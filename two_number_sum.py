def two_number_sum(arr, target_sum):
    """
    :param arr: an array of integers
    :param target_sum: target sum
    :return: two numbers from an array of distinct integers that add to a specified target sum
    """
    nums = {}
    for x in arr:
        y = target_sum - x
        if y in nums:
            return [x, y]
        else:
            nums[x] = True
    return []


card_features = Dict("numbers" = > ("1", "2", "3"),
                                   "colors" = > ("purple", "red", "green"),
                                                "shapes" = > ("oval", "squiggle", "diamond"),
                                                             "shadings" = > ("striped", "solid", "outline"))


# Construct type "Card_" representing each card generated
struct
Card_

number::String
shading::String
color::String
shape::String

end

function
gen_all_cards(card, feature)
"""
This function serves to generate all the possible cards.
:card: struct type, e.g.: "Card_"
:feature: a dictionary of card features
---
:Return: an 1-D array of cards
"""
card_collection = []

for i in Iterators.product(values(feature)...)
    push!(card_collection, card(i...))
end

return card_collection

end

function
compare_cards(cards)
"""
This function serves to compare cards in an array, returning true if and only if all the cards are same or totally different.
:cards: an 1-D array of cards
---
:Return: bool to justify whether the cards are qualified
"""

nbr = [i.number for i in cards]
shd = [i.shading for i in cards]
clr = [i.color for i in cards]
shp = [i.shape for i in cards]

set_length = map(x -> length(Set(x)) == 3 | | length(Set(x)) == 1, [nbr, shd, clr, shp])

if all(set_length)
    return true
else
    return false
end

end

function
sample_cards(cards, num)
"""
This function serves to sample cards from card pool
:cards: an 1-D array of the whole different cards
:num: the number of cards to sample
---
:Return: an 1-D array of the sampled cards /as well as/ the remaining cards from "cards".
"""
r = randperm(length(cards))
sampled_cards, remain_cards = cards[r[1:num]], cards[r[(num + 1):end]]
return sampled_cards, remain_cards

end

function
find_cards(cards;
num = 3)
"""
This function serves to list all the possibilities of combination of cards, C(num, cards), namely.
:cards: an 1-D array of cards
:num: the size of selected cards
---
:Return: all the possible outcomes/combinations
"""
result = []
tmp = Any[""
for _ in 1: num]
l = length(cards)

function
next_card(;
ci = 1, ni = 1)
if ni > num
    push!(result, deepcopy(tmp))
    return
end
for v in ci: l
tmp[ni] = cards[v]
next_card(ci=v + 1, ni=ni + 1)
end
end

next_card()

return result

end

function
game_simulator(cards, n_time;
init_sample_card = 12, add_card = 3)
"""
This function serves to simulate the game running progress.
:cards: an array of all the cards
:n_time: how many times the game will be simulated
:init_sample_card: initial number of sampled cards
:add_card: number of cards added per time
---
:Return: the number of times that the number of cards dealth reaching the number of 18
"""
reaching_counter = 0
all_possible_combination_indeces = find_cards(1:length(cards), num = add_card)
all_possible_combinations = [cards[idx_arr] for idx_arr in all_possible_combination_indeces]
flag_dict = Dict(zip(Set.(all_possible_combinations), map(compare_cards,
                                                          all_possible_combinations)))  # Because the combinations are fixed, we can use a dictionary to record all the possible sets ahead.
print("Local Dictionary Built. \n")
for i in 1: n_time
if i % 100 == 0
    print("Currently Executing Game $(i) \n")
end
card_on_deck = cards  # Reset the cards on deck by copying cards
sampled_cards, card_on_deck = sample_cards(card_on_deck, init_sample_card)
while true
    curnt_len = length(sampled_cards)
    if curnt_len == 18
        reaching_counter += 1
    end
    possible_combination_indeces = find_cards(1:curnt_len, num = add_card)
    possible_combinations = [sampled_cards[idx_arr] for idx_arr in possible_combination_indeces]
    pc_flags = [flag_dict[Set(f)] for f in possible_combinations]
    if length(card_on_deck) < add_card & & length(sampled_cards) == 12
        break
    end
    if any(pc_flags)
        combo_set = possible_combination_indeces[pc_flags][1]
        if curnt_len > 12
            sampled_cards = sampled_cards[setdiff(1:end, combo_set)]
            else
            sampled_cards[combo_set], card_on_deck = sample_cards(card_on_deck, add_card)
        end
    else
        scs, card_on_deck = sample_cards(card_on_deck, add_card)
        sampled_cards = vcat(sampled_cards, scs)
    end
end
end
return reaching_counter, reaching_counter / n_time
end