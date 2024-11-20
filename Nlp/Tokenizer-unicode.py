#!/home/akugyo/Programs/Python/torch/bin/python

def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1

    return counts


def replace_encodings(ids, merge={}):
    merges = {}
    changes_made = False

    for a, b in merge.items():
        new_ids = []
        j = 0

        while j < len(ids) - 1:
            if ids[j] == a and ids[j+1] == b:
                new_ids.append(b)
                j += 1

            else:
                new_ids.append(ids[j])
            j += 1

        if new_ids != ids:
            changes_made = True
            ids = new_ids
            merges[(a, b)] = b

    return (ids, merges)

def encode(text, max_merges=20):
    tokens = [ord(_) for _ in text]
    merge = {}

    for _ in range(max_merges):
        stats = get_stats(tokens)
        pair = min(stats, key=stats.get)
        # print(sorted(((value, key) for key, value in stats.items()), reverse = True))
        print(pair)
        # print(max(stats, key=stats.get))
        if pair in merge:
            break

        new_token = max(merge.values(), default=255) + 1
        merge[pair] = new_token

        tokens, merges = replace_encodings(tokens, merge)
        print(tokens, merges)
        input()

        if not merges:
            break

    return tokens, merge

if __name__ == "__main__":
    text = "Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to “support Unicode” in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. I don’t blame programmers for still finding the whole thing mysterious, even 30 years after Unicode’s inception."
    vocab = {i: chr(i) for i in range(256)}
    print(encode(text))
