# Lua-NeuralNetwork

Single neuron neural network made on Lua, using sigmoid to activation function

```lua
local neural = NeuralNetwork.new(0.1)

for i = 1, 1000 do
  neural:Backpropagate(0.25, 0.75) -- 0.25 -> 0.75
end

print(neural:FeedForward(0.25)) -- getting the value
```
