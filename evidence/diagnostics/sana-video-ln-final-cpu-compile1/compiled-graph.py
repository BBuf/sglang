


def forward(self, L_x_ : torch.Tensor, L_scale_ : torch.Tensor, L_shift_ : torch.Tensor):
    l_x_ = L_x_
    l_scale_ = L_scale_
    l_shift_ = L_shift_
    layer_norm = torch.nn.functional.layer_norm(l_x_, (2240,), None, None, 1e-06);  l_x_ = None
    add = 1 + l_scale_;  l_scale_ = None
    mul = layer_norm * add;  layer_norm = add = None
    add_1 = mul + l_shift_;  mul = l_shift_ = None
    return (add_1,)
    