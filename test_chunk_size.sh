#!/bin/bash

echo "Testing parakeet-mic with different chunk sizes..."
echo ""

# Test with chunk-size < 1 (the problematic case)
echo "Test 1: chunk-size=0.5 (should work now)"
echo "This would previously cause a deadlock"
echo "Command: cargo run --bin parakeet-mic -- --models models --sample-rate 16000 --chunk-size 0.5 --right-context 0.5 --left-context 5 --list-devices"
cargo run --bin parakeet-mic -- --models models --sample-rate 16000 --chunk-size 0.5 --right-context 0.5 --left-context 5 --list-devices

echo ""
echo "Test 2: chunk-size=0.25 (even smaller)"
echo "Command: cargo run --bin parakeet-mic -- --models models --sample-rate 16000 --chunk-size 0.25 --right-context 0.25 --left-context 2 --list-devices"
cargo run --bin parakeet-mic -- --models models --sample-rate 16000 --chunk-size 0.25 --right-context 0.25 --left-context 2 --list-devices

echo ""
echo "Test 3: chunk-size=1.0 (normal case, should still work)"
echo "Command: cargo run --bin parakeet-mic -- --models models --sample-rate 16000 --chunk-size 1.0 --right-context 0.5 --left-context 2 --list-devices"
cargo run --bin parakeet-mic -- --models models --sample-rate 16000 --chunk-size 1.0 --right-context 0.5 --left-context 2 --list-devices

echo ""
echo "All tests completed successfully!"
