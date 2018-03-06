import numpy as np

class node:
    def __init__(self, node1, node2):
        self.left = node1
        self.right = node2
        self.count = node1.count + node2.count


class leaf:
    def __init__(self, tup):
        self.num = tup[0]
        self.count = tup[1]

def houghman(in_codes):
    count_dict = {}
    for val in in_codes.flatten():
        if val not in count_dict:
            count_dict[val] = 0

        count_dict[val] +=1

    list_leafs=[leaf(l) for l in count_dict.items()]
    list_leafs.sort(key=lambda l:l.count)

    while len(list_leafs)>1:
        left_node = list_leafs[0]
        right_node = list_leafs[1]
        new_node = node(left_node,right_node)
        list_leafs = [new_node]+list_leafs[2:]
        list_leafs.sort(key=lambda n:n.count)

    class CodeBuilder:
        def __init__(self):
            self.codes = {}
            # self.reverse_code ={}
            self.buf = ''

        def _DFS(self, in_node):
            if isinstance(in_node, node):
                self.buf+='1'
                self._DFS(in_node.left)
                self.buf = self.buf[:-1]
                self.buf+='0'
                self._DFS(in_node.right)
                self.buf = self.buf[:-1]

            else:
                assert (in_node.num not in self.codes),'Bug in Houghman code trie'
                self.codes[in_node.num]=self.buf
                # assert (self.buf not in self.reverse_code), 'Bug in Houghman code trie'
                # self.reverse_code[self.buf] = in_node.num

        def create_code(self, tree_root):
            self._DFS(tree_root)
            return self.codes

    code_builder = CodeBuilder()
    code_builder.create_code(list_leafs[0])

    return code_builder.codes, list_leafs

def hough_dec(in_str, tree_root):
    ret = []
    curr_pt = tree_root
    for char in in_str:
        if char == '1':
            curr_pt = curr_pt.left
        else:
            curr_pt = curr_pt.right

        if isinstance(curr_pt,leaf):
            ret.append(curr_pt.num)
            curr_pt = tree_root

    return ret

def zigzag_indexs(img_w):
    traversal = []
    direction = 0
    for l in range(img_w):
        traverse_buf = []
        for i in range(l + 1):
            traverse_buf.append([i, l - i])

        if direction == 0:
            traverse_buf.reverse()

        direction = 1 - direction
        traversal.extend(traverse_buf)

    temp = traversal[:-1 * img_w]
    r_traversal = [[img_w - 1 - x[0], img_w - 1 - x[1]] for x in temp]
    r_traversal.reverse()
    traversal.extend(r_traversal)
    return traversal

def zigzag_serialize_str(in_mat, code_dict):
    assert(in_mat.shape[0]==in_mat.shape[1]), 'in_mat is not square!'
    img_w = in_mat.shape[0]
    traversal = zigzag_indexs(img_w)
    serializa_str = ''
    for coor in traversal:
        code=code_dict[in_mat[coor[0],coor[1]]]
        serializa_str = serializa_str+code

    return serializa_str

def zigzag_serialize_list(in_mat):
    assert (in_mat.shape[0] == in_mat.shape[1]), 'in_mat is not square!'
    img_w = in_mat.shape[0]
    traversal = zigzag_indexs(img_w)
    ret_list = []
    for coor in traversal:
        ret_list.append(in_mat[coor[0], coor[1]])

    return ret_list

def flatten_serialize_list(in_mat):
    return list(in_mat.flatten())

def zigzag_deserialize(serialized_str, dec_tree):
    de_code = hough_dec(serialized_str, dec_tree[0])
    img_w =int(np.sqrt(len(de_code)))
    idxs = zigzag_indexs(img_w)
    dec_img = np.zeros((img_w,img_w), np.uint32)
    for idx,coor in enumerate(idxs):
        dec_img[coor[0],coor[1]] = de_code[idx]

    return dec_img


def str_to_byteArray(in_binary_cahr_stream):
    v = int(in_binary_cahr_stream, 2)
    b = bytearray()
    while v:
        b.append(v & 0xff)
        v >>= 8
    return bytes(b[::-1])


def byteArray_to_str(in_byte_arr):
    ret = ''
    for byte in in_byte_arr:
        probe_bit = 1<<7
        for r_shift in range(8):
            ret+='1' if probe_bit&byte else '0'
            probe_bit>>=1

    return ret

class LZW_tree_node:
    def __init__(self, code):
        self.children = {}
        self.code = code

    def add_child(self, input_tok, code):
        assert (input_tok not in self.children), 'Error: {} should not encoded yet following {}'.format(input_tok, self.code)
        self.children[input_tok] = LZW_tree_node(code)


class LZW_dict_tree:
    def __init__(self, int_range):
        self.root = LZW_tree_node(None)


        if len(int_range) == 2:
            int_range = range(int_range[0], int_range[1]+1)

        for code in int_range:
            self.root.add_child(code, code)


def LZW_encode(ls_int, range_int=(0,127)):
    ret = []
    int_range = range(range_int[0], range_int[1]+1)
    lzw_tree = LZW_dict_tree(int_range)
    curr_pt = lzw_tree.root
    next_avail_code = range_int[1]+1

    for in_tok in ls_int:

        if in_tok not in curr_pt.children:
            ret.append(curr_pt.code)
            curr_pt.add_child(in_tok, next_avail_code)
            next_avail_code +=1
            curr_pt = lzw_tree.root.children[in_tok]


        else:
            curr_pt = curr_pt.children[in_tok]

    ret.append(curr_pt.code)

    return ret


def LZW_decode(ls_int, range_int=(0,127)):
    pass

if __name__ == '__main__':
    # code = LZW_encode([0,0,0,1,2,1,0,3,0,1,0,2], (0,3))
    # print (code)
    s = '01101000011010110110100001101001011010000110100101101000011010010110100001101001'

    bytes = str_to_byteArray(s)
    # print (bytes)

    bi_str = byteArray_to_str(bytes)
    print (bi_str)
