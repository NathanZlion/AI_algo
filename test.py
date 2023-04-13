
def num_to_alpha(num):
    assert num > -1, "Number should be positive"
    alphabet_map= lambda num: chr(num+97)

    quotient = num // 26
    remainder = num % 26
    if quotient == 0:
        return alphabet_map(remainder)
    else:
        return num_to_alpha(quotient-1) + alphabet_map(remainder)


