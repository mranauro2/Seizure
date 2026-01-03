"""This file exist to avoid circular import"""
from typing import NamedTuple

NO_AUGMENTATION = 0

class SampleSeizureData(NamedTuple):
    patient_id:str
    file_name:str
    clip_index:int
    has_seizure:bool
    augmentation:int=NO_AUGMENTATION

class NextTimeData(NamedTuple):
    patient_id:str
    file_name:str
    clip_index:int
    file_name_next:str
    clip_index_next:int
    has_seizure:int
    augmentation:int=NO_AUGMENTATION