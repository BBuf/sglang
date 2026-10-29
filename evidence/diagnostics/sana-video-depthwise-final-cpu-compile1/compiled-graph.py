


def forward(self, L_x_ : torch.Tensor, L_self_modules_conv_depth_parameters_bias_ : torch.nn.parameter.Parameter, L_self_modules_conv_depth_parameters_weight_ : torch.nn.parameter.Parameter):
    l_x_ = L_x_
    l_self_modules_conv_depth_parameters_bias_ = L_self_modules_conv_depth_parameters_bias_
    l_self_modules_conv_depth_parameters_weight_ = L_self_modules_conv_depth_parameters_weight_
    hidden_states = torch.conv2d(l_x_, l_self_modules_conv_depth_parameters_weight_, l_self_modules_conv_depth_parameters_bias_, (1, 1), (1, 1), (1, 1), 192);  l_x_ = l_self_modules_conv_depth_parameters_weight_ = l_self_modules_conv_depth_parameters_bias_ = None
    chunk = torch.chunk(hidden_states, 2, dim = 1);  hidden_states = None
    hidden_states_1 = chunk[0]
    gate = chunk[1];  chunk = None
    silu = torch.nn.functional.silu(gate);  gate = None
    mul = hidden_states_1 * silu;  hidden_states_1 = silu = None
    return (mul,)
    