from dataclasses import dataclass
from marshmallow_dataclass import class_schema
import yaml


@dataclass()
class EvaluationPipelineParams:
    path_to_model: str
    path_to_data: str
    path_to_annotations: str
    output_dir: str


EvaluationPipelineParamsSchema = class_schema(EvaluationPipelineParams)


def read_evaluating_pipeline_params(path: str) -> EvaluationPipelineParams:
    with open(path, "r") as input_stream:
        schema = EvaluationPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
