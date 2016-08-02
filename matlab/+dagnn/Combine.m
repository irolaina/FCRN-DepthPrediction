classdef Combine < dagnn.ElementWise

    methods
        function outputs = forward(self, inputs, params)
            %double the size of feature maps, combining four responses
            Y = zeros(size(inputs{1},1)*2, size(inputs{1},2)*2, size(inputs{1},3), size(inputs{1},4), 'like', inputs{1});
            Y(1:2:end, 1:2:end, :, :) = inputs{1};  %A
            Y(2:2:end, 1:2:end, :, :) = inputs{2};  %C
            Y(1:2:end, 2:2:end, :, :) = inputs{3};  %B
            Y(2:2:end, 2:2:end, :, :) = inputs{4};  %D
            outputs{1} = Y;
        end
        
        function [derInputs, derParams] = backward(self, inputs, params, derOutputs)
            %split the feature map into four feature maps of half size
            derInputs{1} = derOutputs{1}(1:2:end, 1:2:end, :, :);
            derInputs{2} = derOutputs{1}(2:2:end, 1:2:end, :, :);
            derInputs{3} = derOutputs{1}(1:2:end, 2:2:end, :, :);
            derInputs{4} = derOutputs{1}(2:2:end, 2:2:end, :, :);
            derParams = {} ;
        end
              
        function outputSizes = getOutputSizes(obj, inputSizes)
            outputSizes{1}(1) = 2*inputSizes{1}(1);
            outputSizes{1}(2) = 2*inputSizes{1}(2);
            outputSizes{1}(3) = inputSizes{1}(3);
            outputSizes{1}(4) = inputSizes{1}(4);
        end
        
        function obj = Combine(varargin)
            obj.load(varargin) ;
        end
    end
end
