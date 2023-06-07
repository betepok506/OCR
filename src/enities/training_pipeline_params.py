from dataclasses import dataclass
from marshmallow_dataclass import class_schema
import yaml
from typing import Tuple


@dataclass()
class TrainingPipelineParams:
    model_name: str
    create_annotations_file: bool
    num_epochs: int
    is_load: bool
    path_to_load: str
    path_to_data: str
    path_to_info_for_model: str
    path_to_annotations: str
    batch_size: int
    output_dir: str
    img_size: Tuple[int, int]
    lr: float


TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)


def read_training_pipeline_params(path: str) -> TrainingPipelineParams:
    with open(path, "r") as input_stream:
        schema = TrainingPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
