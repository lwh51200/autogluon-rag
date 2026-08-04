import unittest

from agrag.modules.agentic.evidence import Evidence, EvidenceStore
from agrag.modules.agentic.planner import QueryPlanner
from agrag.modules.agentic.policy import ActionType, DecisionPolicy
from agrag.modules.agentic.state import AgentState
from agrag.modules.agentic.synthesizer import AnswerSynthesizer
from agrag.modules.agentic.verifier import AnswerVerifier, VerificationLabel


class FakeGenerator:
    def __init__(self, response, model_name="mistral-7b"):
        self.response = response
        self.model_name = model_name
        self.prompts = []

    def generate_response(self, prompt):
        self.prompts.append(prompt)
        return self.response


class ExplodingGenerator:
    """Generator whose calls fail; used to assert it is never invoked."""

    model_name = "mistral-7b"

    def generate_response(self, prompt):  # pragma: no cover - must not be called
        raise AssertionError("generator should not have been called")


class FakeStrandsBackend:
    """Stand-in for StrandsReasoner: returns canned plan/action without Bedrock.

    Mirrors the StrandsReasoner interface used by the planner and policy:
    ``plan_subqueries(query, max)`` and ``choose_action(prompt, legal_values)``.
    """

    def __init__(self, subqueries=None, action=None):
        self._subqueries = subqueries
        self._action = action
        self.plan_calls = []
        self.choose_calls = []

    def plan_subqueries(self, query, max_subqueries):
        self.plan_calls.append((query, max_subqueries))
        return self._subqueries

    def choose_action(self, prompt, legal_values):
        self.choose_calls.append((prompt, list(legal_values)))
        # Honor the enum contract: only return the action if it is legal.
        if self._action in legal_values:
            return self._action
        return None


class ExplodingStrandsBackend:
    """Strands backend whose calls fail; asserts it is never invoked."""

    def plan_subqueries(self, query, max_subqueries):  # pragma: no cover
        raise AssertionError("strands backend should not have been called")

    def choose_action(self, prompt, legal_values):  # pragma: no cover
        raise AssertionError("strands backend should not have been called")


def _store(*texts):
    store = EvidenceStore()
    for i, t in enumerate(texts):
        store.add(Evidence(text=t, doc_id=0, chunk_id=i))
    return store


class TestQueryPlanner(unittest.TestCase):
    def test_simple_query_single_plan(self):
        plan = QueryPlanner().create_plan("what is autogluon")
        self.assertEqual(plan, ["what is autogluon"])

    def test_multi_part_query_split(self):
        plan = QueryPlanner().create_plan("what is autogluon and how does RAG work")
        self.assertEqual(plan[0], "what is autogluon and how does RAG work")
        self.assertIn("what is autogluon", plan)
        self.assertIn("how does RAG work", plan)

    def test_respects_max_subqueries(self):
        planner = QueryPlanner(max_subqueries=2)
        plan = planner.create_plan("a is x and b is y and c is z")
        self.assertLessEqual(len(plan), 2)

    def test_no_split_on_tiny_fragments(self):
        # "cats and dogs" -> parts are 1 word each, not meaningful; no split.
        plan = QueryPlanner().create_plan("cats and dogs")
        self.assertEqual(plan, ["cats and dogs"])


class TestAnswerSynthesizer(unittest.TestCase):
    def test_generate_returns_answer_and_ids(self):
        gen = FakeGenerator("the answer")
        synth = AnswerSynthesizer(gen)
        store = _store("evidence one", "evidence two")
        answer, ids = synth.generate("q", store)
        self.assertEqual(answer, "the answer")
        self.assertEqual(ids, ["e0", "e1"])
        # Prompt includes citations and evidence text.
        self.assertIn("evidence one", gen.prompts[0])
        self.assertIn("doc 0, chunk 0", gen.prompts[0])

    def test_context_token_budget_truncates(self):
        gen = FakeGenerator("ans")
        synth = AnswerSynthesizer(gen, max_context_tokens=3)
        store = _store("one two three four five", "second chunk here")
        texts, ids = synth.build_context(store)
        # First item alone exceeds budget but is always kept; second dropped.
        self.assertEqual(len(texts), 1)
        self.assertEqual(ids, ["e0"])


class TestAnswerVerifier(unittest.TestCase):
    def test_insufficient_evidence_short_circuits(self):
        gen = FakeGenerator("supported")
        verifier = AnswerVerifier(gen, min_evidence_count=2)
        result = verifier.verify("q", "draft", _store("only one"))
        self.assertEqual(result["label"], VerificationLabel.INSUFFICIENT_EVIDENCE.value)
        self.assertFalse(result["is_supported"])
        self.assertEqual(gen.prompts, [])  # model not called

    def test_supported_label(self):
        gen = FakeGenerator("supported")
        verifier = AnswerVerifier(gen, min_evidence_count=2)
        result = verifier.verify("q", "draft", _store("a", "b"))
        self.assertEqual(result["label"], "supported")
        self.assertTrue(result["is_supported"])

    def test_partial_not_matched_as_supported(self):
        gen = FakeGenerator("This is partially_supported by the evidence.")
        verifier = AnswerVerifier(gen, min_evidence_count=1)
        result = verifier.verify("q", "draft", _store("a"))
        self.assertEqual(result["label"], "partially_supported")
        self.assertFalse(result["is_supported"])

    def test_unparseable_defaults_to_unsupported(self):
        gen = FakeGenerator("I have no idea honestly")
        verifier = AnswerVerifier(gen, min_evidence_count=1)
        result = verifier.verify("q", "draft", _store("a"))
        self.assertEqual(result["label"], "unsupported")

    def test_evidence_block_bounded_by_context_budget(self):
        # Two multi-word chunks; a tiny budget must drop the second chunk from
        # the verifier prompt so it does not concatenate all evidence unbounded.
        gen = FakeGenerator("supported")
        verifier = AnswerVerifier(gen, min_evidence_count=2, max_context_tokens=3)
        result = verifier.verify("q", "draft", _store("one two three four five", "second chunk here"))
        # Verifier still runs (enough evidence to satisfy min_evidence_count).
        self.assertEqual(result["label"], "supported")
        prompt = gen.prompts[0]
        self.assertIn("one two three four five", prompt)
        self.assertNotIn("second chunk here", prompt)

    def test_evidence_block_keeps_first_item_over_budget(self):
        # Even when the first chunk alone exceeds the budget, it is kept (so the
        # verifier always has something to judge against).
        gen = FakeGenerator("supported")
        verifier = AnswerVerifier(gen, min_evidence_count=1, max_context_tokens=2)
        result = verifier.verify("q", "draft", _store("a big first chunk with many words"))
        self.assertEqual(result["label"], "supported")
        self.assertIn("a big first chunk with many words", gen.prompts[0])


class TestDecisionPolicy(unittest.TestCase):
    def test_first_action_single_retrieve(self):
        state = AgentState(original_query="q")
        state.plan = ["q"]
        action = DecisionPolicy().next_action(state, EvidenceStore())
        self.assertEqual(action.type, ActionType.RETRIEVE)

    def test_first_action_multi_retrieve_when_plan_has_subqueries(self):
        state = AgentState(original_query="q")
        state.plan = ["q", "sub1", "sub2"]
        action = DecisionPolicy().next_action(state, EvidenceStore())
        self.assertEqual(action.type, ActionType.MULTI_RETRIEVE)
        self.assertEqual(action.args["queries"], ["q", "sub1", "sub2"])

    def test_rewrite_when_evidence_low(self):
        state = AgentState(original_query="q")
        state.record_action("retrieve", tool_name="RetrieveTool")
        action = DecisionPolicy(min_evidence_count=2).next_action(state, _store("only one"))
        self.assertEqual(action.type, ActionType.REWRITE_QUERY)

    def test_abstain_when_low_and_no_rewrites_left(self):
        state = AgentState(original_query="q")
        state.record_action("retrieve", tool_name="RetrieveTool")
        state.record_action("rewrite_query")
        policy = DecisionPolicy(min_evidence_count=2, max_rewrites=1)
        action = policy.next_action(state, _store("only one"))
        self.assertEqual(action.type, ActionType.ABSTAIN)

    def test_rewrite_suppressed_when_no_budget_left(self):
        # Low evidence would normally trigger a rewrite, but with max_iterations
        # set and the loop near its end, there is no budget to act on a rewrite,
        # so the policy abstains instead of wasting the final iteration.
        state = AgentState(original_query="q")
        state.record_action("retrieve", tool_name="RetrieveTool")
        state.iteration = 4  # final iteration of a max_iterations=5 loop
        policy = DecisionPolicy(min_evidence_count=2, max_rewrites=1, max_iterations=5)
        action = policy.next_action(state, _store("only one"))
        self.assertEqual(action.type, ActionType.ABSTAIN)

    def test_rewrite_allowed_when_budget_remains(self):
        # Same low-evidence situation early in the loop still rewrites.
        state = AgentState(original_query="q")
        state.record_action("retrieve", tool_name="RetrieveTool")
        state.iteration = 0
        policy = DecisionPolicy(min_evidence_count=2, max_rewrites=1, max_iterations=5)
        action = policy.next_action(state, _store("only one"))
        self.assertEqual(action.type, ActionType.REWRITE_QUERY)

    def test_draft_when_enough_evidence(self):
        state = AgentState(original_query="q")
        state.record_action("retrieve", tool_name="RetrieveTool")
        action = DecisionPolicy(min_evidence_count=2).next_action(state, _store("a", "b"))
        self.assertEqual(action.type, ActionType.DRAFT_ANSWER)

    def test_accept_verification(self):
        policy = DecisionPolicy()
        self.assertTrue(policy.accept_verification({"label": "supported"}))
        self.assertTrue(policy.accept_verification({"label": "partially_supported"}))
        self.assertFalse(policy.accept_verification({"label": "unsupported"}))
        self.assertFalse(policy.accept_verification({}))


class TestLLMQueryPlanner(unittest.TestCase):
    def test_valid_json_plan(self):
        gen = FakeGenerator('{"subqueries": ["what is autogluon", "how does RAG work"]}')
        planner = QueryPlanner(generator_module=gen, use_llm=True)
        plan = planner.create_plan("explain autogluon and RAG")
        # Original query is always first; LLM subqueries follow, deduped.
        self.assertEqual(plan[0], "explain autogluon and RAG")
        self.assertIn("what is autogluon", plan)
        self.assertIn("how does RAG work", plan)
        self.assertEqual(len(gen.prompts), 1)

    def test_respects_max_subqueries(self):
        gen = FakeGenerator('{"subqueries": ["a b", "c d", "e f", "g h"]}')
        planner = QueryPlanner(max_subqueries=2, generator_module=gen, use_llm=True)
        plan = planner.create_plan("original query here")
        self.assertLessEqual(len(plan), 2)
        self.assertEqual(plan[0], "original query here")

    def test_dedup_of_original_and_repeats(self):
        gen = FakeGenerator('{"subqueries": ["the query", "the query", "extra one"]}')
        planner = QueryPlanner(generator_module=gen, use_llm=True)
        plan = planner.create_plan("the query")
        self.assertEqual(plan, ["the query", "extra one"])

    def test_malformed_json_falls_back_to_rules(self):
        gen = FakeGenerator("not json at all")
        planner = QueryPlanner(generator_module=gen, use_llm=True)
        query = "what is autogluon and how does RAG work"
        plan = planner.create_plan(query)
        # Falls back to the exact rule-based output.
        self.assertEqual(plan, planner._rule_based_plan(query))

    def test_non_list_subqueries_falls_back(self):
        gen = FakeGenerator('{"subqueries": "not a list"}')
        planner = QueryPlanner(generator_module=gen, use_llm=True)
        query = "a is x and b is y"
        self.assertEqual(planner.create_plan(query), planner._rule_based_plan(query))

    def test_non_string_and_empty_elements_filtered(self):
        gen = FakeGenerator('{"subqueries": [123, "", "  ", "good one"]}')
        planner = QueryPlanner(generator_module=gen, use_llm=True)
        plan = planner.create_plan("root query")
        self.assertEqual(plan, ["root query", "good one"])

    def test_json_with_code_fence_tolerated(self):
        gen = FakeGenerator('```json\n{"subqueries": ["sub a", "sub b"]}\n```')
        planner = QueryPlanner(generator_module=gen, use_llm=True)
        plan = planner.create_plan("root")
        self.assertEqual(plan, ["root", "sub a", "sub b"])

    def test_use_llm_without_generator_is_rule_based(self):
        # use_llm=True but no generator -> stays rule-based, no crash.
        planner = QueryPlanner(use_llm=True)
        self.assertEqual(planner.create_plan("what is autogluon"), ["what is autogluon"])

    def test_generator_not_called_when_llm_disabled(self):
        planner = QueryPlanner(generator_module=ExplodingGenerator(), use_llm=False)
        self.assertEqual(planner.create_plan("what is autogluon"), ["what is autogluon"])


class TestLLMDecisionPolicy(unittest.TestCase):
    def _compression_state(self):
        # A state where evidence is sufficient and context is over budget, so the
        # legal set is [COMPRESS_CONTEXT, DRAFT_ANSWER] (the genuine fork).
        state = AgentState(original_query="q")
        state.record_action("retrieve", tool_name="RetrieveTool")
        return state

    def _policy(self, gen):
        return DecisionPolicy(
            min_evidence_count=1,
            use_context_compression=True,
            max_context_tokens=1,
            generator_module=gen,
            use_llm=True,
        )

    def test_llm_chooses_compress(self):
        gen = FakeGenerator('{"action": "compress_context"}')
        policy = self._policy(gen)
        action = policy.next_action(self._compression_state(), _store("aa bb cc", "dd ee ff"))
        self.assertEqual(action.type, ActionType.COMPRESS_CONTEXT)
        # Args are assembled deterministically, not by the model.
        self.assertIn("query", action.args)
        self.assertIn("texts", action.args)

    def test_llm_chooses_draft(self):
        gen = FakeGenerator('{"action": "draft_answer"}')
        policy = self._policy(gen)
        action = policy.next_action(self._compression_state(), _store("aa bb cc", "dd ee ff"))
        self.assertEqual(action.type, ActionType.DRAFT_ANSWER)
        self.assertEqual(action.args, {})

    def test_illegal_choice_falls_back_to_first_legal(self):
        # Model picks an action that is not in the legal set -> deterministic pick.
        gen = FakeGenerator('{"action": "abstain"}')
        policy = self._policy(gen)
        action = policy.next_action(self._compression_state(), _store("aa bb cc", "dd ee ff"))
        self.assertEqual(action.type, ActionType.COMPRESS_CONTEXT)

    def test_garbage_output_falls_back_to_first_legal(self):
        gen = FakeGenerator("I cannot decide")
        policy = self._policy(gen)
        action = policy.next_action(self._compression_state(), _store("aa bb cc", "dd ee ff"))
        self.assertEqual(action.type, ActionType.COMPRESS_CONTEXT)

    def test_single_legal_action_does_not_call_llm(self):
        # Pre-retrieval state: retrieval is forced, so the LLM must not be consulted.
        gen = FakeGenerator('{"action": "draft_answer"}')
        state = AgentState(original_query="q")
        state.plan = ["q"]
        policy = DecisionPolicy(generator_module=gen, use_llm=True)
        action = policy.next_action(state, EvidenceStore())
        self.assertEqual(action.type, ActionType.RETRIEVE)
        self.assertEqual(gen.prompts, [])

    def test_multi_retrieve_forced_and_no_llm_call(self):
        gen = FakeGenerator('{"action": "draft_answer"}')
        state = AgentState(original_query="q")
        state.plan = ["q", "sub1", "sub2"]
        policy = DecisionPolicy(generator_module=gen, use_llm=True)
        action = policy.next_action(state, EvidenceStore())
        self.assertEqual(action.type, ActionType.MULTI_RETRIEVE)
        self.assertEqual(action.args["queries"], ["q", "sub1", "sub2"])
        self.assertEqual(gen.prompts, [])

    def test_llm_disabled_uses_first_legal(self):
        # No generator: multi-option fork resolves to the deterministic first choice.
        policy = DecisionPolicy(min_evidence_count=1, use_context_compression=True, max_context_tokens=1)
        action = policy.next_action(self._compression_state(), _store("aa bb cc", "dd ee ff"))
        self.assertEqual(action.type, ActionType.COMPRESS_CONTEXT)

    # --- widened forks: rewrite-vs-abstain and draft-vs-rewrite ---

    def _low_evidence_state(self):
        state = AgentState(original_query="q")
        state.record_action("retrieve", tool_name="RetrieveTool")
        return state

    def test_low_evidence_llm_can_choose_abstain(self):
        # Low evidence with a rewrite available is now a fork: rewrite OR abstain.
        gen = FakeGenerator('{"action": "abstain"}')
        policy = DecisionPolicy(min_evidence_count=2, max_rewrites=1, generator_module=gen, use_llm=True)
        action = policy.next_action(self._low_evidence_state(), _store("only one"))
        self.assertEqual(action.type, ActionType.ABSTAIN)

    def test_low_evidence_llm_can_choose_rewrite(self):
        gen = FakeGenerator('{"action": "rewrite_query"}')
        policy = DecisionPolicy(min_evidence_count=2, max_rewrites=1, generator_module=gen, use_llm=True)
        action = policy.next_action(self._low_evidence_state(), _store("only one"))
        self.assertEqual(action.type, ActionType.REWRITE_QUERY)
        self.assertIn("query", action.args)

    def test_low_evidence_rule_based_still_rewrites_first(self):
        # Rule-based (no LLM) must keep the original behavior: rewrite is legal[0].
        policy = DecisionPolicy(min_evidence_count=2, max_rewrites=1)
        action = policy.next_action(self._low_evidence_state(), _store("only one"))
        self.assertEqual(action.type, ActionType.REWRITE_QUERY)

    def test_low_evidence_no_rewrite_budget_forces_abstain(self):
        # When no rewrite is possible the branch is single-legal (abstain); the LLM
        # must not be consulted and cannot escape to another action.
        gen = FakeGenerator('{"action": "rewrite_query"}')
        state = self._low_evidence_state()
        state.record_action("rewrite_query")  # exhaust the single rewrite
        policy = DecisionPolicy(min_evidence_count=2, max_rewrites=1, generator_module=gen, use_llm=True)
        action = policy.next_action(state, _store("only one"))
        self.assertEqual(action.type, ActionType.ABSTAIN)
        self.assertEqual(gen.prompts, [])

    def test_sufficient_evidence_llm_can_rewrite_instead_of_draft(self):
        # Enough evidence, nothing failing, rewrite in budget -> draft OR rewrite.
        gen = FakeGenerator('{"action": "rewrite_query"}')
        state = self._low_evidence_state()
        policy = DecisionPolicy(min_evidence_count=2, max_rewrites=1, generator_module=gen, use_llm=True)
        action = policy.next_action(state, _store("a", "b"))
        self.assertEqual(action.type, ActionType.REWRITE_QUERY)

    def test_sufficient_evidence_defaults_to_draft(self):
        # Same state, LLM picks draft; and rule-based (no LLM) also drafts (legal[0]).
        gen = FakeGenerator('{"action": "draft_answer"}')
        state = self._low_evidence_state()
        policy = DecisionPolicy(min_evidence_count=2, max_rewrites=1, generator_module=gen, use_llm=True)
        self.assertEqual(policy.next_action(state, _store("a", "b")).type, ActionType.DRAFT_ANSWER)

        rule_policy = DecisionPolicy(min_evidence_count=2, max_rewrites=1)
        self.assertEqual(rule_policy.next_action(state, _store("a", "b")).type, ActionType.DRAFT_ANSWER)

    def test_sufficient_evidence_no_rewrite_budget_is_single_legal_draft(self):
        gen = FakeGenerator('{"action": "rewrite_query"}')
        state = self._low_evidence_state()
        state.record_action("rewrite_query")  # exhaust rewrites
        policy = DecisionPolicy(min_evidence_count=2, max_rewrites=1, generator_module=gen, use_llm=True)
        action = policy.next_action(state, _store("a", "b"))
        self.assertEqual(action.type, ActionType.DRAFT_ANSWER)
        self.assertEqual(gen.prompts, [])


class TestStrandsQueryPlanner(unittest.TestCase):
    def test_strands_plan_normalized(self):
        # Backend returns raw subqueries; planner prepends original + dedups + caps.
        backend = FakeStrandsBackend(subqueries=["what is autogluon", "how does RAG work"])
        planner = QueryPlanner(strands_backend=backend)
        plan = planner.create_plan("explain autogluon and RAG")
        self.assertEqual(plan[0], "explain autogluon and RAG")
        self.assertIn("what is autogluon", plan)
        self.assertIn("how does RAG work", plan)
        self.assertEqual(len(backend.plan_calls), 1)

    def test_strands_respects_max_subqueries(self):
        backend = FakeStrandsBackend(subqueries=["a b", "c d", "e f", "g h"])
        planner = QueryPlanner(max_subqueries=2, strands_backend=backend)
        plan = planner.create_plan("original query here")
        self.assertLessEqual(len(plan), 2)
        self.assertEqual(plan[0], "original query here")

    def test_strands_dedup_and_non_string_filtered(self):
        backend = FakeStrandsBackend(subqueries=["the query", "the query", 123, "", "  ", "extra one"])
        planner = QueryPlanner(strands_backend=backend)
        plan = planner.create_plan("the query")
        self.assertEqual(plan, ["the query", "extra one"])

    def test_strands_empty_falls_back_to_rules(self):
        backend = FakeStrandsBackend(subqueries=[])
        planner = QueryPlanner(strands_backend=backend)
        query = "what is autogluon and how does RAG work"
        self.assertEqual(planner.create_plan(query), planner._rule_based_plan(query))

    def test_strands_none_falls_back_to_rules(self):
        backend = FakeStrandsBackend(subqueries=None)
        planner = QueryPlanner(strands_backend=backend)
        query = "a is x and b is y"
        self.assertEqual(planner.create_plan(query), planner._rule_based_plan(query))

    def test_strands_backend_failure_falls_back(self):
        planner = QueryPlanner(strands_backend=ExplodingStrandsBackend())
        # The planner swallows the backend error and uses rules.
        self.assertEqual(planner.create_plan("what is autogluon"), ["what is autogluon"])

    def test_strands_takes_precedence_over_llm(self):
        # With both backends configured, Strands wins and the LLM is not called.
        backend = FakeStrandsBackend(subqueries=["from strands"])
        planner = QueryPlanner(generator_module=ExplodingGenerator(), use_llm=True, strands_backend=backend)
        plan = planner.create_plan("root query")
        self.assertEqual(plan, ["root query", "from strands"])

    def test_strands_failure_falls_through_to_llm(self):
        # Strands yields nothing usable -> the raw-LLM path is tried next.
        backend = FakeStrandsBackend(subqueries=None)
        gen = FakeGenerator('{"subqueries": ["llm sub"]}')
        planner = QueryPlanner(generator_module=gen, use_llm=True, strands_backend=backend)
        plan = planner.create_plan("root query")
        self.assertEqual(plan, ["root query", "llm sub"])
        self.assertEqual(len(gen.prompts), 1)


class TestStrandsDecisionPolicy(unittest.TestCase):
    def _compression_state(self):
        state = AgentState(original_query="q")
        state.record_action("retrieve", tool_name="RetrieveTool")
        return state

    def _policy(self, backend, **kw):
        return DecisionPolicy(
            min_evidence_count=1,
            use_context_compression=True,
            max_context_tokens=1,
            strands_backend=backend,
            **kw,
        )

    def test_strands_chooses_compress(self):
        backend = FakeStrandsBackend(action="compress_context")
        policy = self._policy(backend)
        action = policy.next_action(self._compression_state(), _store("aa bb cc", "dd ee ff"))
        self.assertEqual(action.type, ActionType.COMPRESS_CONTEXT)
        # Args assembled deterministically, not by the model.
        self.assertIn("query", action.args)
        self.assertIn("texts", action.args)
        self.assertEqual(len(backend.choose_calls), 1)
        # The backend was offered exactly the legal action values.
        self.assertEqual(backend.choose_calls[0][1], ["compress_context", "draft_answer"])

    def test_strands_chooses_draft(self):
        backend = FakeStrandsBackend(action="draft_answer")
        policy = self._policy(backend)
        action = policy.next_action(self._compression_state(), _store("aa bb cc", "dd ee ff"))
        self.assertEqual(action.type, ActionType.DRAFT_ANSWER)
        self.assertEqual(action.args, {})

    def test_strands_illegal_choice_falls_back_to_first_legal(self):
        # Backend returns an out-of-set action -> None -> deterministic first legal.
        backend = FakeStrandsBackend(action="abstain")
        policy = self._policy(backend)
        action = policy.next_action(self._compression_state(), _store("aa bb cc", "dd ee ff"))
        self.assertEqual(action.type, ActionType.COMPRESS_CONTEXT)

    def test_strands_failure_falls_back_to_first_legal(self):
        policy = self._policy(ExplodingStrandsBackend())
        action = policy.next_action(self._compression_state(), _store("aa bb cc", "dd ee ff"))
        self.assertEqual(action.type, ActionType.COMPRESS_CONTEXT)

    def test_strands_not_consulted_when_single_legal(self):
        # Pre-retrieval: retrieval is forced, so no fork -> backend must not run.
        state = AgentState(original_query="q")
        state.plan = ["q"]
        policy = DecisionPolicy(strands_backend=ExplodingStrandsBackend())
        action = policy.next_action(state, EvidenceStore())
        self.assertEqual(action.type, ActionType.RETRIEVE)

    def test_strands_takes_precedence_over_llm(self):
        # Both backends set: Strands decides, generator is never called.
        backend = FakeStrandsBackend(action="draft_answer")
        policy = DecisionPolicy(
            min_evidence_count=1,
            use_context_compression=True,
            max_context_tokens=1,
            generator_module=ExplodingGenerator(),
            use_llm=True,
            strands_backend=backend,
        )
        action = policy.next_action(self._compression_state(), _store("aa bb cc", "dd ee ff"))
        self.assertEqual(action.type, ActionType.DRAFT_ANSWER)

    def test_strands_failure_falls_through_to_llm(self):
        # Strands yields None -> the raw-LLM path decides the fork.
        backend = FakeStrandsBackend(action=None)
        gen = FakeGenerator('{"action": "draft_answer"}')
        policy = DecisionPolicy(
            min_evidence_count=1,
            use_context_compression=True,
            max_context_tokens=1,
            generator_module=gen,
            use_llm=True,
            strands_backend=backend,
        )
        action = policy.next_action(self._compression_state(), _store("aa bb cc", "dd ee ff"))
        self.assertEqual(action.type, ActionType.DRAFT_ANSWER)
        self.assertEqual(len(gen.prompts), 1)


if __name__ == "__main__":
    unittest.main()
