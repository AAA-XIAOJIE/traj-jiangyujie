"""Reproduce Week 05 from the verified 601-row bundled NGSIM pair."""
import argparse
import hashlib
from importlib.metadata import version
from pathlib import Path
import platform
import time

from pypdf import PdfReader

from .analysis import ROOT, compute, write_json
from .figures import plot_all
from .report import one_page


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"results")
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    begin=time.perf_counter()
    data,test,metrics,traces,parameters,config=compute(args.output)
    plot_all(data,test,metrics,traces,parameters,config,args.output)
    pdf=one_page(args.output,metrics,parameters)
    assert len(PdfReader(pdf).pages)==1,"Expected a one-page course report"
    write_json(args.output/"reproducibility.json",dict(
        input_sha256=hashlib.sha256((ROOT/"data/following_pair.csv").read_bytes()).hexdigest(),
        config_sha256=hashlib.sha256((ROOT/"config.json").read_bytes()).hexdigest(),
        source_code_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/"model").glob("*.py"))},
        environment=dict(python=platform.python_version(),**{name:version(name) for name in ["numpy","pandas","matplotlib","reportlab","pypdf"]}),
        runtime_seconds=round(time.perf_counter()-begin,2),
        output_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.output.iterdir())
                       if p.is_file() and p.name not in ["reproducibility.json","one_page.png"]}))
    print(f"Complete in {time.perf_counter()-begin:.1f} s. One-page report: {pdf}",flush=True)


if __name__=="__main__":main()
