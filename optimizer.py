import torch


def config_opt_schedule(params, args):
    optimizer = torch.optim.Adam(
        params,
        betas=eval(args.adam_betas),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
        lr=args.lr
    )

    if args.lr_schedule == 'fixed':
        return optimizer

    elif args.lr_schedule == 'triangular':
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=0,
            max_lr=args.lr,
            step_size_up=args.warmup_steps,
            step_size_down=args.descend_steps,
            cycle_momentum=False,
        )
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

    else:
        raise ValueError('Unknown lr schedule ' + args.lr_schedule)
