"""List of all modules/functions/classes for documentation.

Note:
    Use "True" for functions and "False" for classes.

"""
MAP = {
    'modules': {
        'act_one': {
            'title': 'single TT-tensor operations',
            'items': {
                'convert': True,
                'copy': True,
                'get': True,
                'get_log': True,
                'get_many': True,
                'get_stab': True,
                'grad': True,
                'interface_ltr': True,
                'interface_rtl': True,
                'mean': True,
                'mean_stab': True,
                'norm': True,
                'norm_stab': True,
                'sum': True,
                'sum_stab': True,
            },
        },
        'act_two': {
            'title': 'operations with a pair of TT-tensors',
            'items': {
                'accuracy': True,
                'add': True,
                'mul': True,
                'mul_scalar': True,
                'mul_scalar_stab': True,
                'sub': True,
            },
        },
        'als': {
            'title': 'construct TT-tensor by TT-ALS',
            'items': {
                'als': True,
            },
        },
        'cross': {
            'title': 'construct TT-tensor by TT-cross',
            'items': {
                'cross': True,
            },
        },
        'data': {
            'title': 'functions for working with datasets',
            'items': {
                'accuracy_on_data': True,
            },
        },
        'maxvol': {
            'title': 'compute the maximal-volume submatrix',
            'items': {
                'maxvol': True,
                'maxvol_rect': True,
            },
        },
        'sample': {
            'title': 'random sampling for/from the TT-tensor',
            'items': {
                'sample': True,
                'sample_lhs': True,
                'sample_rand': True,
            }
        },
        'svd': {
            'title': 'SVD-based algorithms for matrices and tensors',
            'items': {
                'matrix_skeleton': True,
                'svd': True,
            },
        },
        'tensors': {
            'title': 'collection of explicit useful TT-tensors',
            'items': {
                'rand': True,
                'rand_norm': True,
            },
        },
        'transformation': {
            'title': 'orthogonalization, truncation and other transformations of the TT-tensors',
            'items': {
                'full': True,
                'orthogonalize_rtl': True,
                'orthogonalize_rtl_stab': True,
            },
        },
        'vis': {
            'title': 'visualization methods for tensors',
            'items': {
                'show': True,
            },
        },
    },
}
