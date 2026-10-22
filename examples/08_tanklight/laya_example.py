"""Laya on its own: sort one message with yes/no questions. Runs offline after the first download."""
from laya import Router

router = Router()  # picks the English or multilingual model per message

questions = {
    "water":  {"type": "noul", "instructions": "Is the household out of drinking water or is their tank low?"},
    "sewage": {"type": "noul", "instructions": "Is the toilet, drain or sewage tank full or backing up?"},
    "health": {"type": "noul", "instructions": "Does the water smell, taste or look bad, or did someone get sick?"},
    "urgency": {"type": "score", "instructions": "How urgent is it?", "criteria": ["routine", "today", "emergency"]},
}

for message in ["Our water tank is almost empty and we have a baby at home.",
                "Le réservoir d'eaux usées est plein, pouvez-vous venir vider?"]:
    result = router.predict(message, questions)
    answers = result["answers"]
    print(message)
    print("  model:", result["routing"]["model"])
    for name in ("water", "sewage", "health"):
        print(f"  {name:<7} P(yes) = {answers[name]['noul']:.2f}")
    print(f"  urgency score = {answers['urgency']['score']:.2f}  (0 routine, 1 today, 2 emergency)")
