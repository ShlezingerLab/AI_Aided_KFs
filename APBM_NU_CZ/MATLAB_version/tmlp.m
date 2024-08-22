classdef tmlp < handle
    % Tales MLP. 
    % function obj = tmlp(input_dim, output_dim, hidden_layer_sizes, act_funcs) 
    %
    % example: tnn = tmlp(2, 1, [10, 5]) 
    %
    %
   
    properties
        input_dim               % input dimension
        output_dim              % output dimension
        params                  % vector with all parameters
        act_funcs               % activation functions e.g., ['relu','sigmoid', 'linear']
        nlayers                 % number of layers
        nparams                 % number of parameters
        nparams_per_layer       % number of parameters per layer
        Wshapes                 % cell with shape of weight matrices per layer
        Ws                      % cell with weight matrix per layer
        bs                      % cell with bias per layer
    end
        
    methods 
%         function obj = tmlp(input_dim, output_dim, hidden_layer_sizes, act_funcs) 
        function obj = tmlp(input_dim, output_dim, hidden_layer_sizes) 
            
            r = 1e-1;
%             r = 1e-3;
            obj.input_dim = input_dim;
            obj.output_dim = output_dim;
            obj.nlayers = length(hidden_layer_sizes)+1;
            % nbiasparams = nlayers
            % input_dim * 
            
            obj.nparams_per_layer = [input_dim, hidden_layer_sizes].*[hidden_layer_sizes, output_dim] + 1;
            obj.nparams = sum(obj.nparams_per_layer);
            
%             obj.act_funcs = act_funcs;
            obj.Ws = cell(obj.nlayers,1);
            obj.Wshapes = cell(obj.nlayers,1);
            obj.bs= cell(obj.nlayers,1);
            obj.Ws{1} = r * randn(hidden_layer_sizes(1), input_dim);
            obj.bs{1} = r * randn;
            obj.Wshapes{1} = [hidden_layer_sizes(1), input_dim];
            for i=2:obj.nlayers-1
                obj.Ws{i} = r * randn(hidden_layer_sizes(i), hidden_layer_sizes(i-1));
                obj.bs{i} = r * randn;
                obj.Wshapes{i} = [hidden_layer_sizes(i), hidden_layer_sizes(i-1)];
            end
            obj.Ws{end} = r * randn(output_dim, hidden_layer_sizes(end));
            obj.bs{end} = r * randn;
            obj.Wshapes{end} = [output_dim, hidden_layer_sizes(end)];
        end        
        
        
        function y=forward(obj, x)
            y=x;
            for i=1:obj.nlayers - 1
                y = obj.Ws{i} * y + obj.bs{i};
                y = relu(y);
%                 y = obj.act_funcs{i}(y);
            end
            % linear activation function!
            y = obj.Ws{i+1} * y + obj.bs{i+1};
        end
        
        function set_params(obj, params)
            c = 1;
            for i=1:obj.nlayers
                layer_par = params(c:c+obj.nparams_per_layer(i)-1);
                obj.Ws{i} = reshape(layer_par(1:end-1), obj.Wshapes{i}(1), obj.Wshapes{i}(2));
                obj.bs{i} = layer_par(end);
                c = c + obj.nparams_per_layer(i);
            end

        end
        function params = get_params(obj)
            params = zeros(obj.nparams,1);
            c = 1;
            for i=1:obj.nlayers
                params(c:c+obj.nparams_per_layer(i)-1) = [obj.Ws{i}(:); obj.bs{i}];
                c = c + obj.nparams_per_layer(i);
            end
        end
    end
end

function y = relu(x)
    y = max(0, x);
end