#!/home/akugyo/Programs/Python/PyTorch/bin/python

def n_grams(text, n):
    """
    Takes Tokens or Text, returns a list of n-grams
    """

    return [text[i: i+n] for i in range(len(text) - n + 1)]


input_1 = ["mary", ",", "n't", "slap", "green", "witch", "."]
print(n_grams(input_1, 3))

input_2 = "Baby this is what you came for, Lightning strikes every times you"
print(n_grams(input_2, 3))
