from sympy import hyper
import yaml


def get_hyp_data(hyp):
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    return hyp



cfg = get_hyp_data('./config.yaml')
print(cfg)
