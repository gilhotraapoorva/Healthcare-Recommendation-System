import json

def process_evidences():
    with open("data//release_evidences.json") as f:
        data = json.loads(f.read())

    binary_evidence = {}
    other_evidence = {}

    for evidence in data.keys():
        code_question = data[evidence]["code_question"]
        if code_question in binary_evidence.keys():
            if data[evidence]["data_type"] == "B":
                binary_evidence[code_question].append(evidence)
            else:
                other_evidence[code_question].append(evidence)
        else:
            if data[evidence]["data_type"] == "B":
                binary_evidence[code_question] = [evidence, ]
                other_evidence[code_question] = []
            else:
                binary_evidence[code_question] = []
                other_evidence[code_question] = [evidence, ]

    evidences = []
    initial_evidence = None
    for code_question in binary_evidence.keys():
        evidence = binary_evidence[code_question][0]
        print(data[evidence]["question_en"], "(y/n)")
        cmd = input()
        if cmd == "y":
            if initial_evidence is None:
                initial_evidence = data[evidence]["name"]
            evidences.append(data[evidence]["name"])

            for evidence in other_evidence[code_question]:
                print(data[evidence]["question_en"])
                data_type = data[evidence]["data_type"]
                if data_type == "M":
                    for i in range(len(data[evidence]["possible-values"])):
                        possible_value = data[evidence]["possible-values"][i]
                        print(f"{i})", data[evidence]["value_meaning"][possible_value]["en"])
                    selections = list(map(int, input().split(",")))
                    for i in selections:
                        value = data[evidence]["possible-values"][i]
                        if value != "nulle_part":
                            evidences.append(f"{code_question}_@_{value}")
                elif data_type == "C":
                    possible_values = list(map(str, data[evidence]['possible-values']))
                    print(f"Possible Values: {''.join(possible_values)}")

                    value = int(input())
                    evidences.append(f"{code_question}_@_{value}")

            print("Anything else! (y/n)")
            cmd = input()
            if cmd == "n":
                break

    return evidences

print(process_evidences())
