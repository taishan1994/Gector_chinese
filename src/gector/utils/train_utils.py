from transformers import AdamW, get_linear_schedule_with_warmup


def build_optimizer_and_scheduler(args, model, t_total, freeze_embedding=False):
    module = (
        model.module if hasattr(model, "module") else model
    )

    if freeze_embedding:
        embedding_name_list = ('embeddings.word_embeddings.weight',
                               'embeddings.position_embeddings.weight',
                               'embeddings.token_type_embeddings.weight')
        for named_para in model.named_parameters():
            named_para[1].requires_grad = False if named_para[
                                                       0] in embedding_name_list else True

    # 差分学习率
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    model_params = list(module.named_parameters())
    optimizer_grouped_parameters = [{
        'params': [
            p for n, p in model_params
            if not any(nd in n for nd in no_decay)
        ],
        'weight_decay':
            0.01
    }, {
        'params':
            [p for n, p in model_params if any(nd in n for nd in no_decay)],
        'weight_decay':
            0.0
    }]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_proportion * t_total),
        num_training_steps=t_total
    )

    return optimizer, scheduler
