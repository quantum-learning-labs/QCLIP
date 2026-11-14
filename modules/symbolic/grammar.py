from lambeq.text2diagram import CCGType 
import re

def get_type(ccgtype):
    res_arr = []
    type_arr = [ccgtype]
    while type_arr: 
        cur_type = type_arr.pop()
        if cur_type.is_over:
            type_arr.append(cur_type.result)
            if cur_type.argument.is_complex: 
                old_arg = cur_type.argument
                new_arg = CCGType(result=old_arg.argument, direction=old_arg.direction, argument=old_arg.result)
                type_arr.append(new_arg)
            else:
                type_arr.append(cur_type.argument)
        elif cur_type.is_under: 
            if cur_type.argument.is_complex:
                old_arg = cur_type.argument
                new_arg = CCGType(result=old_arg.argument, direction=old_arg.direction, argument=old_arg.result)
                type_arr.append(new_arg)
            else:
                type_arr.append(cur_type.argument)
            type_arr.append(cur_type.result)
        else: 
            res_arr.append(cur_type.name)
    return res_arr[::-1]

def strip_sent(caption):
    caption = re.sub(r'[^\w\s]', '', caption)
    return " ".join(caption.split()).lower()
