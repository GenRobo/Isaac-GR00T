from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

ur5e_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["front", "hand"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "single_arm",
            "gripper",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),
        modality_keys=[
            "single_arm",
            "gripper",
        ],
        action_configs=[
            # single_arm: 6 UR5e joints - use RELATIVE for smoother actions
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # gripper: use ABSOLUTE (binary/continuous open-close)
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
}

register_modality_config(ur5e_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)