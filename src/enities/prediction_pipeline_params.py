from dataclasses import dataclass
from marshmallow_dataclass import class_schema
import yaml


@dataclass()
class PredictingPipelineParams:
    model_name: str
    path_to_model: str
    path_to_info_for_model: str
    path_to_data: str
    path_to_annotations: str
    create_annotations: bool
    img_size: tuple
    batch_size: int
    output_dir: str


PredictingPipelineParamsSchema = class_schema(PredictingPipelineParams)


def read_predicting_pipeline_params(path: str) -> PredictingPipelineParams:
    with open(path, "r") as input_stream:
        schema = PredictingPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
