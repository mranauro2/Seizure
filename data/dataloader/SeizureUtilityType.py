"""This file exist to avoid circular import"""
from typing import NamedTuple

NO_AUGMENTATION = 0

class SampleData(NamedTuple):
    patient_id:str
    file_name:str
    clip_index:int
    has_seizure:bool
    augmentation:int=NO_AUGMENTATION