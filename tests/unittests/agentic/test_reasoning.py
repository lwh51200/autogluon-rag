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


if __name__ == "__main__":
    unittest.main()
