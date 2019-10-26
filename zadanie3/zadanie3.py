import json
import numpy as np
import glob


def load_summary(test_nr):
    with open('./input/' + test_nr + '/summary.txt') as summary:
        room_list = summary.read().splitlines()
        return room_list


def rooms_list(test_nr):
    room_list = load_summary(test_nr)
    rooms = []
    for record in room_list:
        room = record.split('.jpg, ')
        rooms.append(room)
    return rooms


def rooms_to_dict():
    rooms = np.array(rooms_list(test_nr))
    dictionary = dict(zip(rooms[:,0], rooms[:,1]))
    return dictionary


def describe(filename):
    room_name = rooms_to_dict()[filename[:-9]]
    descriptions = dict()
    word_list = []
    with open('./input/' + test_nr + '/' + filename) as json_file:
        data = json.load(json_file)
        for word in data:
            word_list.append(word['name'].lower())
    descriptions[room_name] = word_list
    return descriptions


def description_to_file(test_nr, filename):
    with open('./descriptions/descriptions_' + test_nr[-1] + '.txt', 'a+') as file:  # Use file to refer to the file object
        file.write(str(describe(filename)) + "\n")


def save_all_descriptions(test_nr):
    for filepath in glob.glob("./input/" + test_nr + "/*.json"):
        filename = filepath.split("\\")[1]
        description_to_file(test_nr, filename)


def vocabulary_append(test_nr):
    vocabulary = set()
    for filepath in glob.glob("./input/" + test_nr + "/*.json"):
        filename = filepath.split("\\")[1]
        descriptions = describe(filename)
        for word in descriptions[rooms_to_dict()[filename[:-9]]]:
            vocabulary.add(word)
    return vocabulary


def vocabulary_to_file(test_nr):
    vocabulary = vocabulary_append(test_nr)
    with open('./vocabularies/vocabulary_' + test_nr[-1] + '.txt', 'w') as file:  # Use file to refer to the file object
        file.write(str(vocabulary))


test_nr = "test_1" # nazwa folderu testowego, ktorego tworzymy descriptions i vocabulary (test_# i # nalezy do (1, 2, 3))
save_all_descriptions(test_nr)  # zapisuje descriptions do ./descriptions/descriptions_#.txt gdzie #-nr folderu testowego
vocabulary_to_file(test_nr)  # zapisuje vocabulary do ./vocabulary/vocabulary_#.txt gdzie #-nr folderu testowego
