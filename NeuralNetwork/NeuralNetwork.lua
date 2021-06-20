--[[
  @author univb
  @since February 17th, 2021
  @desc Single neuron neural network made on Lua, using sigmoid as activation function
  
  @constructor
  -- NeuralNetwork.new(learningRate || 0.1: Float)

  @methods
  -- NeuralNetwork:FeedForward(input: Number): float
  -- NeuralNetwork:Backpropagate(input: Number, target: Number): void
]]
-- * Setting seed
math.randomseed(os.time() * (os.clock() * 1000))

-- * Functions
local function sigmoid(x)
  return 1 / (1 + math.exp(-x))
end

local function dsigmoid(x)
  return x * (1 - x)
end

-- * Class
local NeuralNetwork = {}
NeuralNetwork.__index = NeuralNetwork


function NeuralNetwork.new(lr)
  local self = setmetatable({LearningRate = lr or 0.1}, NeuralNetwork)

  self._weights = {
    w0 = math.random(),
    w1 = math.random()
  }
  self._biases = {
    b0 = math.random(),
    b1 = math.random()
  }
  

  return self
end


function NeuralNetwork:FeedForward(input)
  local hidden = sigmoid((input * self._weights.w0) + self._biases.b0)
  local output = sigmoid((hidden * self._weights.w1) + self._biases.b1)

  return output
end


function NeuralNetwork:Backpropagate(input, target)
  local hidden = sigmoid((input * self._weights.w0) + self._biases.b0)
  local output = sigmoid((hidden * self._weights.w1) + self._biases.b1)

  local output_error = target - output
  local gradient_hidden = (output_error * dsigmoid(output)) * self.LearningRate

  self._weights.w1 = (self._weights.w1 + (hidden * gradient_hidden))
  self._biases.b1 = self._biases.b1 + gradient_hidden

  local hidden_error = self._weights.w0 * output_error
  local gradient_input = (hidden_error * dsigmoid(hidden)) * self.LearningRate

  self._weights.w0 = (self._weights.w0 + (input * gradient_input))
  self._biases.b0 = self._biases.b1 + gradient_input
end


return NeuralNetwork
