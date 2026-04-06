from roboflow import Roboflow

rf = Roboflow(api_key="0b8MWbeUWhJLTOlcHvNC")

project = rf.workspace("roboflow-universe-projects").project("hard-hat-workers")
dataset = project.version(1).download("yolov8")

print("Dataset downloaded!")