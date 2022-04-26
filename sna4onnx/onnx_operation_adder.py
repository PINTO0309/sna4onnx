#! /usr/bin/env python

import os
import sys
import traceback
from argparse import ArgumentParser
import onnx
import onnx_graphsurgeon as gs
import numpy as np
from typing import Optional
import sog4onnx

class Color:
    BLACK          = '\033[30m'
    RED            = '\033[31m'
    GREEN          = '\033[32m'
    YELLOW         = '\033[33m'
    BLUE           = '\033[34m'
    MAGENTA        = '\033[35m'
    CYAN           = '\033[36m'
    WHITE          = '\033[37m'
    COLOR_DEFAULT  = '\033[39m'
    BOLD           = '\033[1m'
    UNDERLINE      = '\033[4m'
    INVISIBLE      = '\033[08m'
    REVERCE        = '\033[07m'
    BG_BLACK       = '\033[40m'
    BG_RED         = '\033[41m'
    BG_GREEN       = '\033[42m'
    BG_YELLOW      = '\033[43m'
    BG_BLUE        = '\033[44m'
    BG_MAGENTA     = '\033[45m'
    BG_CYAN        = '\033[46m'
    BG_WHITE       = '\033[47m'
    BG_DEFAULT     = '\033[49m'
    RESET          = '\033[0m'

AVAILABLE_DTYPES = [
    'float32',
    'float64',
    'int32',
    'int64',
    'str',
]

DTYPES_TO_ONNX_DTYPES = {
    float: onnx.TensorProto.FLOAT,
    int: onnx.TensorProto.INT64,
    str: onnx.TensorProto.STRING,
}

DTYPES_TO_NUMPY_TYPES = {
    'float32': np.float32,
    'float64': np.float64,
    'int32': np.int32,
    'int64': np.int64,
}

NUMPY_TYPES_TO_ONNX_DTYPES = {
    np.dtype('float32'): onnx.TensorProto.FLOAT,
    np.dtype('float64'): onnx.TensorProto.DOUBLE,
    np.dtype('int32'): onnx.TensorProto.INT32,
    np.dtype('int64'): onnx.TensorProto.INT64,
}


def add(
    connection_src_op_name: str,
    connection_dist_op_name: str,
    add_op_type: str,
    add_op_input_variables: Optional[dict] = None,
    add_op_output_variables: Optional[dict] = None,
    add_op_attributes: Optional[dict] = None,
    input_onnx_file_path: Optional[str] = '',
    onnx_graph: Optional[onnx.ModelProto] = None,
    output_onnx_file_path: Optional[str] = '',
    non_verbose: Optional[bool] = False,
) -> onnx.ModelProto:
    """
    Parameters
    ----------
    connection_src_op_name: str
        Specify the name of the output OP from which to connect.\n\n\
        e.g. \n\
        [OpA] outnameA - inpnameB [OpB] outnameB - inpnameC [OpC]\n\
        When extrapolating a new OP between OpA and OpB.\n\
        --connection_src_op_name outnameA\n\n\
        This need not be specified only when the type of the newly added OP is Constant.

    connection_dist_op_name: str
        Specify the name of the output OP from which to connect.\n\
        e.g.\n\n\
        [OpA] outnameA - inpnameB [OpB] outnameB - inpnameC [OpC]\n\
        When extrapolating a new OP between OpA and OpB.\n\
        --connection_dist_op_name inpnameB

    add_op_type: str
        ONNX op type.\n\
        See below for the types of OPs that can be specified.\n\n\
        e.g. "Add", "Div", "Gemm", ...\n\
        https://github.com/onnx/onnx/blob/main/docs/Operators.md

    add_op_input_variables: Optional[dict]
        Specify input variables for the OP to be generated.\n\
        See below for the variables that can be specified.\n\n\
        {"input_var_name1": [numpy.dtype, shape], "input_var_name2": [dtype, shape], ...}\n\n\
        e.g. add_op_input_variables = {"name1": [np.float32, [1,224,224,3]], "name2": [np.bool_, [0]], ...}\n\
        https://github.com/onnx/onnx/blob/main/docs/Operators.md

    add_op_output_variables: Optional[dict]
        Specify output variables for the OP to be generated.\n\
        See below for the variables that can be specified.\n\n\
        {"output_var_name1": [numpy.dtype, shape], "output_var_name2": [dtype, shape], ...}\n\n\
        e.g. add_op_output_variables = {"name1": [np.float32, [1,224,224,3]], "name2": [np.bool_, [0]], ...}\n\
        https://github.com/onnx/onnx/blob/main/docs/Operators.md

    add_op_attributes: Optional[dict]
        Specify output add_op_attributes for the OP to be generated.\n\
        See below for the add_op_attributes that can be specified.\n\n\
        {"attr_name1": value1, "attr_name2": value2, "attr_name3": value3, ...}\n\n\
        e.g. add_op_attributes = {"alpha": 1.0, "beta": 1.0, "transA": 0, "transB": 0}\n\
        Default: None\n\
        https://github.com/onnx/onnx/blob/main/docs/Operators.md

    input_onnx_file_path: Optional[str]
        Input onnx file path.\n\
        Either input_onnx_file_path or onnx_graph must be specified.\n\
        Default: ''

    onnx_graph: Optional[onnx.ModelProto]
        onnx.ModelProto.\n\
        Either input_onnx_file_path or onnx_graph must be specified.\n\
        onnx_graph If specified, ignore input_onnx_file_path and process onnx_graph.

    output_onnx_file_path: Optional[str]
        Output onnx file path. If not specified, no ONNX file is output.\n\
        Default: ''

    non_verbose: Optional[bool]
        Do not show all information logs. Only error logs are displayed.\n\
        Default: False

    Returns
    -------
    changed_graph: onnx.ModelProto
        Changed onnx ModelProto.
    """

    # Unspecified check for input_onnx_file_path and onnx_graph
    if not input_onnx_file_path and not onnx_graph:
        print(
            f'{Color.RED}ERROR:{Color.RESET} '+
            f'One of input_onnx_file_path or onnx_graph must be specified.'
        )
        sys.exit(1)

    if add_op_type not in ['Constant', 'ConstantOfShape']:
        if not add_op_input_variables:
            print(
                f'{Color.RED}ERROR:{Color.RESET} '+
                f'If add_op_type is other than Const or ConstantOfShape, '+
                f'add_op_input_variables must be specified.'
            )
            sys.exit(1)

    # Loading Graphs
    # onnx_graph If specified, onnx_graph is processed first
    if not onnx_graph:
        onnx_graph = onnx.load(input_onnx_file_path)
    graph = gs.import_onnx(onnx_graph)
    graph.cleanup().toposort()

    # Obtaining the opset of the original ONNX file
    opset = graph.opset

    # Generate an ONNX graph that holds only one OP
    single_op_graph = sog4onnx.generate(
        op_type = add_op_type,
        opset = opset,
        input_variables = add_op_input_variables,
        output_variables = add_op_output_variables,
        attributes = add_op_attributes,
        non_verbose = True,
    )


    ###
    ### Main Logic
    ###


    # Shape Estimation
    changed_graph = None
    try:
        changed_graph = onnx.shape_inference.infer_shapes(gs.export_onnx(graph))
    except:
        changed_graph = gs.export_onnx(graph)
        if not non_verbose:
            print(
                f'{Color.YELLOW}WARNING:{Color.RESET} '+
                'The input shape of the next OP does not match the output shape. '+
                'Be sure to open the .onnx file to verify the certainty of the geometry.'
            )

    # Save
    if output_onnx_file_path:
        onnx.save(changed_graph, output_onnx_file_path)

    if not non_verbose:
        print(f'{Color.GREEN}INFO:{Color.RESET} Finish!')

    # Return
    return changed_graph


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--input_onnx_file_path',
        type=str,
        required=True,
        help='Input onnx file path.'
    )
    parser.add_argument(
        '--connection_src_op_name',
        type=str,
        help=\
            'Specify the name of the output OP from which to connect. \n'+
            'e.g. \n'+
            '[OpA] outnameA - inpnameB [OpB] outnameB - inpnameC [OpC] \n'+
            'When extrapolating a new OP between OpA and OpB. \n'+
            '--connection_src_op_name outnameA \n'+
            'This need not be specified only when the type of the newly added OP is Constant.'
    )
    parser.add_argument(
        '--connection_dist_op_name',
        type=str,
        required=True,
        help=\
            'Specify the name of the output OP from which to connect. \n'+
            'e.g. \n'+
            '[OpA] outnameA - inpnameB [OpB] outnameB - inpnameC [OpC] \n'+
            'When extrapolating a new OP between OpA and OpB. \n'+
            '--connection_dist_op_name inpnameB'
    )
    # https://github.com/onnx/onnx/blob/main/docs/Operators.md
    parser.add_argument(
        '--add_op_type',
        type=str,
        required=True,
        default='',
        help=\
            'ONNX OP type. \n'+
            'https://github.com/onnx/onnx/blob/main/docs/Operators.md'
    )
    parser.add_argument(
        '--add_op_input_variables',
        type=str,
        nargs=3,
        action='append',
        help=\
            'input_variables can be specified multiple times. \n'+
            '--input_variables variable_name numpy.dtype shape \n'+
            'https://github.com/onnx/onnx/blob/main/docs/Operators.md \n\n'+
            'e.g.\n'+
            '--input_variables i1 float32 [1,3,5,5] \n'+
            '--input_variables i2 int32 [1] \n'+
            '--input_variables i3 float64 [1,3,224,224]'
    )
    parser.add_argument(
        '--add_op_output_variables',
        type=str,
        nargs=3,
        action='append',
        help=\
            'output_variables can be specified multiple times. \n'+
            '--output_variables variable_name numpy.dtype shape \n'+
            'https://github.com/onnx/onnx/blob/main/docs/Operators.md \n\n'+
            'e.g.\n'+
            '--output_variables o1 float32 [1,3,5,5] \n'+
            '--output_variables o2 int32 [1] \n'+
            '--output_variables o3 float64 [1,3,224,224]'
    )
    parser.add_argument(
        '--add_op_attributes',
        nargs=3,
        action='append',
        help=\
            'attributes can be specified multiple times. \n'+
            '--attributes name dtype value \n'+
            'dtype is one of "float32" or "float64" or "int32" or "int64" or "str". \n'+
            'https://github.com/onnx/onnx/blob/main/docs/Operators.md \n\n'+
            'e.g.\n'+
            '--attributes alpha float32 1.0 \n'+
            '--attributes beta float32 1.0 \n'+
            '--attributes transA int64 0 \n'+
            '--attributes transB int64 0'
    )
    parser.add_argument(
        '--output_onnx_file_path',
        type=str,
        default='',
        help=\
            'Output onnx file path. \n'+
            'If not specified, a file with "_mod" appended '+
            'to the end of input_onnx_file_path is output.  \n'+
            'e.g. aaa.onnx -> aaa_mod.onnx'
    )
    parser.add_argument(
        '--non_verbose',
        action='store_true',
        help='Do not show all information logs. Only error logs are displayed.'
    )
    args = parser.parse_args()

    # add op input variables
    """
    input_variables_tmp = {'name': [dtype, shape]}
    """
    input_variables_tmp = None
    if args.add_op_input_variables:
        input_variables_tmp = {input_variable[0]: [getattr(np, input_variable[1]), eval(input_variable[2])] for input_variable in args.add_op_input_variables}

    # add op output variables
    """
    output_variables_tmp = {'name': [dtype, shape]}
    """
    output_variables_tmp = None
    if args.add_op_output_variables:
        output_variables_tmp = {output_variable[0]: [getattr(np, output_variable[1]), eval(output_variable[2])] for output_variable in args.add_op_output_variables}

    # add op attributes
    """
    attributes_tmp = {'name': value}
    """
    attributes_tmp = None
    if args.add_op_attributes:

        if args.add_op_type in ['Constant','ConstantOfShape']:
            if len(args.add_op_attributes) > 1:
                print(
                    f'{Color.RED}ERROR:{Color.RESET} '+
                    f'Only one attribute may be specified for Constant and ConstantOfShape.'
                )
                sys.exit(1)

        attributes_tmp = {}
        for attribute in args.add_op_attributes:
            # parse
            attr_name = attribute[0]
            attr_type = attribute[1]
            attr_value = eval(attribute[2])

            # dtype check
            if attr_type not in AVAILABLE_DTYPES:
                print(
                    f'{Color.RED}ERROR:{Color.RESET} '+
                    f'The dtype that can be specified for attributes is one of the {AVAILABLE_DTYPES}. \n'+
                    f'dtype:{attr_type}'
                )
                sys.exit(1)

            # Conversion from python types to numpy types
            # However, only if the input values are in list format
            # Constant(value), ConstantOfShape
            if (args.add_op_attributes == 'Constant' and attr_name in ['sparse_value', 'value']) or (args.add_op_attributes == 'ConstantOfShape'):
                if isinstance(attr_value, list):
                    attr_value = np.asarray(attr_value, dtype=DTYPES_TO_NUMPY_TYPES[attr_type])

            attributes_tmp[attr_name] = attr_value

    # output_onnx_file_path
    output_onnx_file_path = ''
    if args.output_onnx_file_path:
        output_onnx_file_path = args.output_onnx_file_path
    else:
        output_onnx_file_path = f'{os.path.splitext(args.input_onnx_file_path)[0]}_mod.onnx'

    # Load
    onnx_graph = onnx.load(args.input_onnx_file_path)

    # OP add
    changed_graph = add(
        input_onnx_file_path=None,
        onnx_graph=onnx_graph,
        connection_src_op_name=args.connection_src_op_name,
        connection_dist_op_name=args.connection_dist_op_name,
        add_op_type=args.add_op_type,
        add_op_input_variables=input_variables_tmp,
        add_op_output_variables=output_variables_tmp,
        add_op_attributes=attributes_tmp,
        output_onnx_file_path=output_onnx_file_path,
        non_verbose=args.non_verbose,
    )


if __name__ == '__main__':
    main()