import argparse
from lncrnapy.scripts.classify import args as c_args, description as c_desc
from lncrnapy.scripts.train import args as t_args, description as t_desc
from lncrnapy.scripts.pretrain import args as p_args, description as p_desc
from lncrnapy.scripts.embeddings import args as e_args, description as e_desc
from lncrnapy.scripts.bpe import args as b_args, description as b_desc

script_args = {
    'classify': (c_args, c_desc),
    'train': (t_args, t_desc),
    'pretrain': (p_args, p_desc),
    'embeddings': (e_args, e_desc),
    'bpe': (b_args, b_desc)
}

for script in script_args:

    args, desc = script_args[script]
    p = argparse.ArgumentParser()
    for arg in args:
        p.add_argument(arg, **args[arg])
    
    text = f'{desc}\n\n::\n\n\tpython -m ' \
           f'lncnrapy.scripts.{script}{p.format_usage().split(".py")[1]}\n' \
           f'\n**Positional arguments:**'

    arg_descriptors = (p.format_help().split('positional arguments:')[1]
                       .split('optional arguments:'))
    pos, opt = arg_descriptors[0], arg_descriptors[1]
    
    fmt_pos = ''
    for line in pos.split('\n'):
        if len(line) > 0:
            line = line.strip().split(' ')
            line = f'\n  `{line[0]}`\n' + " ".join(line[1:])
            fmt_pos += line
    text += fmt_pos

    text += '\n\n**Optional arguments**'
    fmt_opt = ''
    for line in opt.split('\n'):
        if len(line) > 0:
            line = line.strip()
            if line.startswith('-h, --help'):
                line = f'\n  `-h, \-\-help`\n    Show help message.'
            elif line.startswith('--'):
                line = line.split(' ')
                if len(line) == 1:
                    line = f'\n  `\-\-{line[0].strip("--")}`'
                elif len(line) == 2:
                    line = f'\n  `\-\-{line[0].strip("--")}` {line[1]}'
                else:
                    line = f'\n  `\-\-{line[0].strip("--")}` {line[1]}' \
                           f'\n    {" ".join(line[2:]).strip()}'
            else:
                line = f'\n    {line}'
            fmt_opt += line
    text += fmt_opt
    
    file = open(f'docs/script_args/{script}.rst', 'w')
    file.write(text)
    file.close()