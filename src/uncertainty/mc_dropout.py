import torch


def enable_dropout(model):
    """
    Enable dropout layers during inference.
    """
    for module in model.modules():
        if module.__class__.__name__.startswith("Dropout"):
            module.train()


def mc_dropout_predict(
    model,
    spatial_x,
    freq_x,
    passes: int = 20
):

    enable_dropout(model)

    predictions = []

    with torch.no_grad():
        for _ in range(passes):
            logits = model(spatial_x, freq_x)
            probs = torch.sigmoid(logits)
            predictions.append(probs)

    preds = torch.stack(predictions)
    mean_prob = preds.mean()
    std_prob = preds.std()

    confidence = 1 - std_prob.item()

    return {
        "mean_probability": mean_prob.item(),
        "uncertainty_std": std_prob.item(),
        "confidence_score": confidence
    }
