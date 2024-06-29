import numpy as np
from argparse import ArgumentParser
import tqdm

# BE SURE TO RUN `python easy_redis.py --just-start` from the DataPortraits repo
import dataportraits


infer_by_stride = True


def init_sketch(path_to_portrait="mmarone/quip_20200301.25-1.bf"):
    try:
        portrait = dataportraits.from_hub(path_to_portrait, verbose=True, overwrite=False)
    except Exception as e:
        # already loaded, hopefully
        portrait = dataportraits.RedisBFSketch(host="localhost", port=8899, key=path_to_portrait.split("/")[-1], width=25)
    return portrait


def get_all(string: list, portrait) -> dict:
    output = portrait.contains_from_text([string])[0]
    chains = []
    for i, bool_segment in enumerate(output["is_member"]):
        if bool_segment:
            chains.append(output["segments"][i:i+25])
    return chains


def check(list_to_check: list, portrait) -> dict:
    if infer_by_stride:
        output = portrait.contains_from_text(list_to_check, sort_chains_by_length=True, infer_by_stride=True)
    else:
        output = portrait.contains_from_text(list_to_check)
    if not len(output):
        return {"avg_percent_overlap": None}
    
    if len([report['segments'] for report in output]) == 0:
        percent_overlap = 0
    else:
        is_members = [sum(report['is_member']) for report in output]
        segments = [len(report['segments']) for report in output]
        overlap_list = [is_member / segment if segment != 0 else 0 for is_member, segment in zip(is_members, segments)]
        percent_overlap = np.mean(overlap_list)

    return {
        "avg_percent_overlap": percent_overlap,
        "percent_overlap_list": overlap_list,
    }


if __name__ == "__main__":
    # NOTE: below line will test it
    portrait = init_sketch()
    string = """everal men, including McMaster, appeared in court, were found guilty of rioting but acquitted of manslaughter, and served several months in prison. Ten years after returning to Canada, McMaster confessed to the killing. As he had already been found not guilty of manslaughter, he """
    print(check([string], portrait))

