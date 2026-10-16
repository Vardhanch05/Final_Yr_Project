import re
import ast
from typing import List, Dict, Any, Tuple

class MinorityProtectionGate:
    """
    Inference-Time Minority Protection Gate for Multi-Agent Debate (ReConcile - ACL 2024).
    
    Prevents Conformity Bias & Sycophancy Collapse:
    When a minority agent holds the correct answer in Round 1, standard multi-agent debate
    frequently causes the minority agent to cave under majority peer pressure in Round 2.
    
    This gate intercepts split rounds, verifies the algorithmic/arithmetic steps of the
    minority response, and protects grounded dissent from being overwritten.
    """
    def __init__(self, arithmetic_check: bool = True, protection_threshold: float = 0.8):
        self.arithmetic_check = arithmetic_check
        self.protection_threshold = protection_threshold

    def extract_equations(self, text: str) -> List[Tuple[str, float]]:
        """
        Extracts step-by-step arithmetic statements like '12 + 8 = 20' or '5 * 4 = 20'.
        """
        equations = []
        # Match patterns like: <num> <op> <num> = <num>
        pattern = r'(\d+(?:\.\d+)?)\s*([\+\-\*\/])\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)'
        matches = re.findall(pattern, text)
        for a, op, b, res in matches:
            expr_str = f"{a} {op} {b}"
            equations.append((expr_str, float(res)))
        return equations

    def verify_arithmetic_grounding(self, reasoning: str) -> Tuple[float, int, int]:
        """
        Verifies whether the math steps inside the reasoning chain are valid.
        Returns: (grounding_score, verified_steps, total_steps)
        """
        equations = self.extract_equations(reasoning)
        if not equations:
            # If no explicit math equations found, return neutral grounding
            return 0.7, 0, 0
        
        valid_steps = 0
        for expr, claimed_res in equations:
            try:
                # Safely evaluate simple arithmetic
                tree = ast.parse(expr, mode='eval')
                actual_res = eval(compile(tree, '<string>', 'eval'))
                if abs(actual_res - claimed_res) < 1e-4:
                    valid_steps += 1
            except Exception:
                pass
        
        grounding_score = valid_steps / len(equations)
        return grounding_score, valid_steps, len(equations)

    def evaluate_debate_round(
        self,
        agents_responses: List[Dict[str, Any]],
        gold_answer: str,
        round_num: int = 1
    ) -> Dict[str, Any]:
        """
        Processes a round of debate among N agents (e.g. Agent A, Agent B, Agent C).
        
        Each response dict contains:
            - 'agent_id': str
            - 'answer': str (e.g. '42' or 'yes')
            - 'reasoning': str
            - 'confidence_level': float
        """
        votes: Dict[str, List[int]] = {}
        for idx, resp in enumerate(agents_responses):
            ans = str(resp.get('answer', '')).strip().lower()
            votes.setdefault(ans, []).append(idx)

        total_agents = len(agents_responses)
        majority_ans = max(votes.keys(), key=lambda k: len(votes[k]))
        is_split = len(votes) > 1

        gate_action = "NO_INTERVENTION"
        protected_agent_idx = None
        arbitrated_answer = majority_ans
        grounding_scores = []

        # Analyze grounding for all agents
        for idx, resp in enumerate(agents_responses):
            score, valid, total = self.verify_arithmetic_grounding(resp.get('reasoning', ''))
            grounding_scores.append({
                'agent_id': resp.get('agent_id', f'Agent_{idx}'),
                'score': score,
                'valid_steps': valid,
                'total_steps': total,
                'ans': str(resp.get('answer', '')).strip().lower()
            })

        if is_split:
            # Check if any minority agent has superior grounding
            majority_indices = votes[majority_ans]
            minority_answers = [ans for ans in votes if ans != majority_ans]

            for min_ans in minority_answers:
                min_indices = votes[min_ans]
                # Compare average grounding between minority and majority
                min_grounding = sum(grounding_scores[i]['score'] for i in min_indices) / len(min_indices)
                maj_grounding = sum(grounding_scores[i]['score'] for i in majority_indices) / len(majority_indices)

                # If minority has verified arithmetic and exceeds majority grounding
                if min_grounding >= self.protection_threshold and min_grounding > maj_grounding:
                    gate_action = "PROTECTED_MINORITY"
                    arbitrated_answer = min_ans
                    protected_agent_idx = min_indices[0]
                    break

        # Standard ReConcile consensus (pure majority vote)
        naive_consensus = majority_ans

        return {
            "round": round_num,
            "is_split": is_split,
            "naive_majority_ans": naive_consensus,
            "gate_arbitrated_ans": arbitrated_answer,
            "gold_answer": str(gold_answer).strip().lower(),
            "gate_action": gate_action,
            "naive_correct": naive_consensus == str(gold_answer).strip().lower(),
            "gate_correct": arbitrated_answer == str(gold_answer).strip().lower(),
            "grounding_analysis": grounding_scores
        }
