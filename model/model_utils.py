from torch import Tensor
def freeze(module) -> None:
    """
    Freezes module's parameters.
    """
    for parameter in module.parameters():
        parameter.requires_grad = False


def get_freezed_parameters(module) -> list[Tensor]:
    """
    Returns names of freezed parameters of the given module.
    """
    freezed_parameters = []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            freezed_parameters.append(name)

    return freezed_parameters



