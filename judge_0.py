from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np
from fuzzywuzzy import fuzz

@dataclass
class Constraint:
    name:str
    weight:float
    description:str

class LogicScore:
    def __init__(self, constraints:List[Constraint], label_map: Dict[str, float]):
        """
        Initialize the LogicScore object.
        Args:
            constraints: List of Constraints objects
            label_map: Dict mapping labels to scores (e.g, {"High": 1.0, "Medium": 0.5, "Low": 0.2})
        """
        self.constraints = constraints
        self.label_map = label_map

    def _get_fuzzy_score(self, response:str)->float:
        """
        Match the response against label_map keys using fuzzy matching.
        Returns the score of the best matching label.
        """
        best_label=None
        best_ratio=0

        #Normalize response
        clean_response=response.strip()

        for label in self.label_map.keys():
            #Simple ratio match
            ratio=fuzz.ratio(clean_response.lower(), label.lower())
            if ratio>best_ratio:
                best_ratio=ratio
                best_label=label
            
        #Threshold
        if best_ratio>65 and best_label:
            return self.label_map[best_label]
        
        return 0.0
    
    def score(self, labels: List[str])->Dict[str, Any]:
        """
        Compute the weighted score for a list of labels
        """
        if len(labels)!=len(self.constraints):
            raise ValueError(f"Label count ({len(labels)} != Constraint count ({len(self.constraints)}))")
        
        #1. Map Labels to Scores (Fuzzy Match)
        scores=[self._get_fuzzy_score(label) for label in labels]

        #2. Get Weights
        weights=[c.weight for c in self.constraints]

        #3. Hadamard Product
        weighted_scores=np.multiply(scores, weights)

        #4. Final Score
        final_score=float(np.sum(weighted_scores))

        #5. Normalize
        max_label_score=max(self.label_map.values()) if self.label_map else 1.0
        total_weight=sum(weights)
        max_possible_score=total_weight*max_label_score

        normalized_score=(final_score/max_possible_score)*100 if max_possible_score>0 else 0.0

        return {
            "final_score":final_score,
            "normalized_score": round(normalized_score,2),
            "breakdown": [
                {
                    "constraint":c.name,
                    "weight":c.weight,
                    "label":l,
                    "score":s,
                    "weighted_score":float(ws)
                }
                for c, l, s, w, ws in zip(self.constraints, labels, scores, weights, weighted_scores)
            ]

        }
    
    def tune_weights(self, breakdown: List[Dict[str, Any]], target_score: float, alpha: float = 0.01) -> List[Constraint]:
        """
        Adjust weights using Gradient Descent
            Args:
                breakdown: 'breakdown' list from score() result.
                target_score: Ground truth score from end user.
                alpha: Learning rate.
            Returns:
                List of updated Constraint objects.
        """
        #1. Calculate Error Term
        
        current_raw_score=sum(item['weighted_score'] for item in breakdown)

        # We need to convert target_score to raw target
        max_label_val=max(self.label_map.values()) if self.label_map else 1.0
        total_weight=sum(c.weight for c in self.constraints)
        max_possible=total_weight*max_label_val

        #target raw = (target%/100)*max_possible
        target_raw=target_score*max_possible/100

        #delta
        delta=target_raw-current_raw_score

        updated_constraints=[]

        #2. Update Weights
        for i, item in enumerate(breakdown):
            c=self.constraints[i]
            x_i=item['score']

            #Update Rule: w_new = w_old + alpha * delta * input
            #Normalise alpha by dividing by number of constraints to keep gradient stable
            weight_change = (alpha / len(self.constraints)) * delta * x_i

            new_weight=c.weight+weight_change

            #Clip weights
            new_weight=max(0.0, min(2.0, new_weight))

            #Update Constraint object
            c.weight=round(new_weight, 4)
            updated_constraints.append(c)
        
        return updated_constraints

