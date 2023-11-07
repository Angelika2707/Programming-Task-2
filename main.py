from project_classes import UserInput, InteriorPointMethod, SimplexMethod


def round_with_accuracy(value, accuracy, number_of_decimal_places_accuracy):
    if number_of_decimal_places_accuracy == 0:
        if str(value)[-2:] == ".5":
            if value > 0:
                return int(value + 0.5)
            elif value < 0:
                return int(value - 0.5)
        else:
            return int(round(value))
    elif abs(value) <= accuracy:
        return 0
    elif '.' in str(value):
        if number_of_decimal_places_accuracy < len(str(value).split('.')[1]):
            if str(value).split('.')[1][number_of_decimal_places_accuracy] == "5":
                if value > 0:
                    return round(value + 5 * 10 ** (-number_of_decimal_places_accuracy - 1),
                                 number_of_decimal_places_accuracy)
                elif value < 0:
                    return round(value - 5 * 10 ** (-number_of_decimal_places_accuracy - 1),
                                 number_of_decimal_places_accuracy)
    return round(value, number_of_decimal_places_accuracy)


user_input = UserInput()
user_input.collect_data()
simplex_method = SimplexMethod(user_input.size, user_input.C, user_input.A, user_input.b, user_input.a)
result = simplex_method.revised_simplex_method()
if result is None:
    print("The method is not applicable!")
else:
    if user_input.a != 1:
        num_of_decimal_places_accuracy = len(str(user_input.a).split('.')[1])
    else:
        num_of_decimal_places_accuracy = 0
    rounded_x = [round_with_accuracy(x, user_input.a, num_of_decimal_places_accuracy) for x in result[1:]]
    rounded_z = round_with_accuracy(result[0], user_input.a, num_of_decimal_places_accuracy)
    print("Optimal solution was found by Simplex method:")
    print("A vector of decision variables - x*: (" + ', '.join([f"{x:.{num_of_decimal_places_accuracy}f}" for x
                                                                in rounded_x]) + ")")
    if user_input.max_problem:
        print(f"Maximum value of the objective function z: {rounded_z:.{num_of_decimal_places_accuracy}f}")
    else:
        print(f"Maximum value of the objective function z: {(-1) * rounded_z:.{num_of_decimal_places_accuracy}f}")

user_input.additional_input_for_interior_point()
method_interior_point_1 = InteriorPointMethod(user_input.size, user_input.C, user_input.A, user_input.b, user_input.a,
                                              0.5, user_input.x)
result_1 = method_interior_point_1.interior_point_method()
if result_1 is None:
    print("Interior-point method (alpha = 0.5):")
    print("The problem does not have solution!")
else:
    if user_input.a != 1:
        num_of_decimal_places_accuracy = len(str(user_input.a).split('.')[1])
    else:
        num_of_decimal_places_accuracy = 0
    rounded_x = [round_with_accuracy(x, user_input.a, num_of_decimal_places_accuracy) for x in result_1[1:]]
    rounded_z = round_with_accuracy(result_1[0], user_input.a, num_of_decimal_places_accuracy)
    print("Optimal solution was found by Interior-point method (alpha = 0.5):")
    print("A vector of decision variables - x*: (" + ', '.join([f"{x:.{num_of_decimal_places_accuracy}f}" for x
                                                                in rounded_x]) + ")")
    if user_input.max_problem:
        print(f"Maximum value of the objective function z: {rounded_z:.{num_of_decimal_places_accuracy}f}")
    else:
        print(f"Maximum value of the objective function z: {(-1) * rounded_z:.{num_of_decimal_places_accuracy}f}")

method_interior_point_2 = InteriorPointMethod(user_input.size, user_input.C, user_input.A, user_input.b, user_input.a,
                                              0.9, user_input.x)
result_2 = method_interior_point_2.interior_point_method()

if result_2 is None:
    print("Interior-point method (alpha = 0.9):")
    print("The problem does not have solution!")
else:
    if user_input.a != 1:
        num_of_decimal_places_accuracy = len(str(user_input.a).split('.')[1])
    else:
        num_of_decimal_places_accuracy = 0
    rounded_x = [round_with_accuracy(x, user_input.a, num_of_decimal_places_accuracy) for x in result_2[1:]]
    rounded_z = round_with_accuracy(result_2[0], user_input.a, num_of_decimal_places_accuracy)
    print("Optimal solution was found by Interior-point method (alpha = 0.9):")
    print("A vector of decision variables - x*: (" + ', '.join([f"{x:.{num_of_decimal_places_accuracy}f}" for x
                                                                in rounded_x]) + ")")
    if user_input.max_problem:
        print(f"Maximum value of the objective function z: {rounded_z:.{num_of_decimal_places_accuracy}f}")
    else:
        print(f"Maximum value of the objective function z: {(-1) * rounded_z:.{num_of_decimal_places_accuracy}f}")