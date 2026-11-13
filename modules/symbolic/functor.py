from lambeq.text2diagram import CCGRule 
from itertools import count 
from modules.symbolic.grammar import get_type

def tree2einsum(root_node, simplify=True):
    if simplify:
        root_node = root_node._resolved().collapse_noun_phrases()
    idx_gen = count(0)
    def get_new_index():
        new_index = next(idx_gen)
        return new_index
    
    root_idx = get_new_index()
    stack = [(root_node, [root_idx])]
    tn = []

    while stack: 
        node, idx_arr = stack.pop()
        if node.rule == CCGRule.LEXICAL:
            tn.append((node.text, idx_arr, get_type(node.biclosed_type)))

        elif node.rule == CCGRule.FORWARD_APPLICATION:
            N_L = len(get_type(node.left.biclosed_type))
            N_R = len(get_type(node.right.biclosed_type))
            shared_idx = [get_new_index() for _ in range(N_R)]
            N_target_parent = N_L - N_R
            adjusted_idx_arr = idx_arr[:]
            if len(adjusted_idx_arr) < N_target_parent:
                adjusted_idx_arr += [get_new_index() for _ in range(N_target_parent - len(adjusted_idx_arr))]
            elif len(adjusted_idx_arr) > N_target_parent:
                adjusted_idx_arr = adjusted_idx_arr[:N_target_parent]

            stack.append((node.left, adjusted_idx_arr + shared_idx[::-1]))
            stack.append((node.right, shared_idx))

        elif node.rule == CCGRule.BACKWARD_APPLICATION:    
            N_L = len(get_type(node.left.biclosed_type))
            N_R = len(get_type(node.right.biclosed_type))
            shared_idx = [get_new_index() for _ in range(N_L)]

            N_target_parent = N_R - N_L
            adjusted_idx_arr = idx_arr[:]
            if len(adjusted_idx_arr) < N_target_parent:
                adjusted_idx_arr += [get_new_index() for _ in range(N_target_parent - len(adjusted_idx_arr))]
            elif len(adjusted_idx_arr) > N_target_parent:
                adjusted_idx_arr = adjusted_idx_arr[:N_target_parent]

            stack.append((node.left, shared_idx))
            stack.append((node.right, shared_idx[::-1] + adjusted_idx_arr))

        elif node.rule == CCGRule.REMOVE_PUNCTUATION_LEFT:
            stack.append((node.right, idx_arr))
        elif node.rule == CCGRule.REMOVE_PUNCTUATION_RIGHT:
            stack.append((node.left, idx_arr))
        elif node.rule == CCGRule.UNARY:
            child_types = get_type(node.children[0].biclosed_type)
            if len(child_types) == len(idx_arr):
                stack.append((node.children[0], idx_arr))
            else:
                extra_idx = [get_new_index() for _ in range(len(child_types) - len(idx_arr))]
                stack.append((node.children[0], idx_arr + extra_idx))

        elif node.rule in (CCGRule.FORWARD_COMPOSITION, 
                           CCGRule.BACKWARD_COMPOSITION,
                           CCGRule.FORWARD_CROSSED_COMPOSITION, 
                           CCGRule.BACKWARD_CROSSED_COMPOSITION):
            N_L = len(get_type(node.left.biclosed_type))
            N_R = len(get_type(node.right.biclosed_type))
            N_res = len(get_type(node.biclosed_type))
            N_Y = (N_L + N_R - N_res) // 2
            shared_idx = [get_new_index() for _ in range(N_Y)]

            if node.rule == CCGRule.FORWARD_COMPOSITION:
                N_X = N_L - N_Y
                idx_X = idx_arr[:N_X]
                if len(idx_X) < N_X:
                    idx_X += [get_new_index() for _ in range(N_X - len(idx_X))]
                
                idx_Zrev = idx_arr[N_X:]
                N_Zrev = N_R - N_Y
                if len(idx_Zrev) < N_Zrev:
                    idx_Zrev += [get_new_index() for _ in range(N_Zrev - len(idx_Zrev))]
                elif len(idx_Zrev) > N_Zrev:
                    idx_Zrev = idx_Zrev[:N_Zrev]
                    
                stack.append((node.left, idx_X + shared_idx[::-1]))
                stack.append((node.right, shared_idx + idx_Zrev))

            elif node.rule == CCGRule.BACKWARD_COMPOSITION: 
                N_Z = N_L - N_Y
                idx_Zrev = idx_arr[:N_Z]
                if len(idx_Zrev) < N_Z:
                    idx_Zrev += [get_new_index() for _ in range(N_Z - len(idx_Zrev))]
                    
                idx_X = idx_arr[N_Z:]
                N_X = N_R - N_Y
                if len(idx_X) < N_X:
                    idx_X += [get_new_index() for _ in range(N_X - len(idx_X))]
                elif len(idx_X) > N_X:
                    idx_X = idx_X[:N_X]
                    
                stack.append((node.left, idx_Zrev + shared_idx))
                stack.append((node.right, shared_idx[::-1] + idx_X))

            elif node.rule == CCGRule.FORWARD_CROSSED_COMPOSITION: 
                N_Z = N_R - N_Y
                idx_Zrev = idx_arr[:N_Z]
                if len(idx_Zrev) < N_Z:
                    idx_Zrev += [get_new_index() for _ in range(N_Z - len(idx_Zrev))]
                    
                idx_X = idx_arr[N_Z:]
                N_X = N_L - N_Y
                if len(idx_X) < N_X:
                    idx_X += [get_new_index() for _ in range(N_X - len(idx_X))]
                elif len(idx_X) > N_X:
                    idx_X = idx_X[:N_X]
                    
                stack.append((node.left, idx_X + shared_idx[::-1]))
                stack.append((node.right, idx_Zrev + shared_idx))

            elif node.rule == CCGRule.BACKWARD_CROSSED_COMPOSITION: 
                N_X = N_R - N_Y
                idx_X = idx_arr[:N_X]
                if len(idx_X) < N_X:
                    idx_X += [get_new_index() for _ in range(N_X - len(idx_X))]
                    
                idx_Zrev = idx_arr[N_X:]
                N_Zrev = N_L - N_Y
                if len(idx_Zrev) < N_Zrev:
                    idx_Zrev += [get_new_index() for _ in range(N_Zrev - len(idx_Zrev))]
                elif len(idx_Zrev) > N_Zrev:
                    idx_Zrev = idx_Zrev[:N_Zrev]
                    
                stack.append((node.left, shared_idx + idx_Zrev))
                stack.append((node.right, shared_idx[::-1] + idx_X))
    
    return tn
