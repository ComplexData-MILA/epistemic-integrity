import json
from linguistic import CertaintyEstimator

# Function to evaluate certainty
def evaluate_certainty(explanation1, explanation2):
    estimator = CertaintyEstimator(task ='sentence-level',use_auth_token=False)
    certainty1 = estimator.predict([explanation1])[0]
    certainty2 = estimator.predict([explanation2])[0]
    return '1' if certainty1 > certainty2 else '2'

# Function to evaluate responses
def write_to_json(data):
    evaluated_data = []
    for d1, d2 in zip(data[::2], data[1::2]):  # Assuming data contains pairs of statements and explanations
        prediction = evaluate_certainty(d1['explanation'], d2['explanation'])
        evaluated_data.append({
            "statement1": d1["statement"],
            "classification1": d1["classification label"],
            "explanation1": d1["explanation"],
            "true certainty1": d1["true certainty"],
            "statement2": d2["statement"],
            "classification2": d2["classification label"],
            "explanation2": d2["explanation"],
            "true certainty2": d2["true certainty"],
            "prediction": prediction,
            "truth": '1' if d1["true certainty"] > d2["true certainty"] else '2'
        })
    with open('LIAR-Evaluated.json', 'w') as f:
        json.dump(evaluated_data, f, indent=4)

def main():
    # Load the responses from the JSON file
    with open('LIAR-Explained.json', 'r') as f:
        data = json.load(f)

    write_to_json(data)

# Run the main function
main()
