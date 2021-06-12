-- Example
local NeuralNetwork = require('NeuralNetwork')

local neural = NeuralNetwork.new(0.1)

-- Setting up epochs
local epochs = 1000

-- Training our neural network
for i = 1, epochs do
  neural:Backpropagate(0.25, 0.8) -- for example, 0.1 -> 0.8
end

-- Getting our response
local response = neural:FeedForward(0.1)

print(response)