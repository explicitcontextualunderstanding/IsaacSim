// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*
Test is implemented using the doctest C++ testing framework:
  https://github.com/doctest/doctest/blob/master/doc/markdown/readme.md
*/

#include <carb/BindingsUtils.h>
#include <carb/logging/Log.h>

#include <doctest/doctest.h>
#include <isaacsim/core/includes/ScopedCudaDevice.h>
#include <isaacsim/ucx/core/UcxListener.h>
#include <isaacsim/ucx/core/UcxListenerRegistry.h>
#include <ucxx/api.h>

#include <chrono>
#include <memory>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

using namespace isaacsim::ucx::core;
using namespace isaacsim::core::includes;

// Helper class to create client endpoint for testing
// Uses a shared worker (from UCXListener) instead of creating its own,
// following the official UCXX pattern where both endpoints share the same worker.
class TestClient
{
public:
    std::shared_ptr<ucxx::Worker> worker;
    std::shared_ptr<ucxx::Endpoint> endpoint;

    TestClient(std::shared_ptr<ucxx::Worker> sharedWorker, const std::string& ip, uint16_t port) : worker(sharedWorker)
    {
        endpoint = worker->createEndpointFromHostname(ip, port, true);
        // Give the endpoint a moment to establish connection
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Convenience methods that forward to endpoint
    std::shared_ptr<ucxx::Request> tagSend(const void* buffer, size_t length, ucxx::Tag tag)
    {
        return endpoint->tagSend(const_cast<void*>(buffer), length, tag);
    }

    std::shared_ptr<ucxx::Request> tagReceive(void* buffer, size_t length, ucxx::Tag tag, ucxx::TagMask mask)
    {
        return endpoint->tagRecv(buffer, length, tag, mask);
    }

    std::shared_ptr<ucxx::Request> tagMultiSend(const std::vector<const void*>& buffer,
                                                const std::vector<size_t>& size,
                                                const std::vector<int>& isCuda,
                                                ucxx::Tag tag)
    {
        std::vector<void*> non_const_buffer;
        for (const void* ptr : buffer)
        {
            non_const_buffer.push_back(const_cast<void*>(ptr));
        }
        return endpoint->tagMultiSend(non_const_buffer, size, isCuda, tag, false);
    }

    std::shared_ptr<ucxx::Request> tagMultiReceive(ucxx::Tag tag, ucxx::TagMask mask)
    {
        return endpoint->tagMultiRecv(tag, mask, false);
    }

    ~TestClient()
    {
        // Just reset the endpoint - the worker is shared with the listener
        // and its progress thread is managed by the listener's lifecycle
        if (endpoint)
        {
            endpoint.reset();
        }
    }
};

TEST_SUITE("isaacsim.ucx.core.listener_registry_tests")
{
    TEST_CASE("UCXListenerRegistry: basic registry operations")
    {
        // Clean up any existing listeners before starting
        UCXListenerRegistry::shutdown();

        // Test creating a listener with addListener (ephemeral port)
        auto listener1 = UCXListenerRegistry::addListener();
        REQUIRE(listener1.get() != nullptr);

        uint16_t port1 = listener1->getPort();
        CHECK(port1 != 0);
        CHECK(UCXListenerRegistry::isListenerRegistered(port1));

        // Test getting the same listener again using addListener with specific port
        auto listener1_again = UCXListenerRegistry::addListener(port1);
        REQUIRE(listener1_again.get() == listener1.get());

        // Test creating a second listener with different ephemeral port using addListener
        auto listener2 = UCXListenerRegistry::addListener();
        REQUIRE(listener2.get() != nullptr);

        uint16_t port2 = listener2->getPort();
        CHECK(port2 != 0);
        CHECK(port2 != port1);
        CHECK(UCXListenerRegistry::isListenerRegistered(port2));

        // Test creating another listener with addListener
        auto listener3 = UCXListenerRegistry::addListener();
        REQUIRE(listener3.get() != nullptr);

        uint16_t port3 = listener3->getPort();
        CHECK(port3 != 0);
        CHECK(port3 != port1);
        CHECK(port3 != port2);

        // Test creating a listener on a specific port
        auto listener4 = UCXListenerRegistry::addListener(23456);
        REQUIRE(listener4.get() != nullptr);

        CHECK(listener4->getPort() == 23456);
        CHECK(UCXListenerRegistry::isListenerRegistered(23456));

        // Test removing a listener
        UCXListenerRegistry::removeListener(port1);
        CHECK_FALSE(UCXListenerRegistry::isListenerRegistered(port1));
        CHECK(UCXListenerRegistry::isListenerRegistered(port2));
        CHECK(UCXListenerRegistry::isListenerRegistered(port3));
        CHECK(UCXListenerRegistry::isListenerRegistered(23456));

        // Test shutdown clears all listeners
        UCXListenerRegistry::shutdown();
        CHECK_FALSE(UCXListenerRegistry::isListenerRegistered(port2));
        CHECK_FALSE(UCXListenerRegistry::isListenerRegistered(port3));
        CHECK_FALSE(UCXListenerRegistry::isListenerRegistered(23456));
    }

    TEST_CASE("UCXListenerRegistry: listener communication functionality")
    {
        // Clean up any existing listeners before starting
        UCXListenerRegistry::shutdown();
        //
        // Create a listener through the registry using addListener for ephemeral port
        auto registry_listener = UCXListenerRegistry::addListener();
        REQUIRE(registry_listener.get() != nullptr);

        uint16_t server_port = registry_listener->getPort();
        CHECK(server_port != 0);
        CHECK(UCXListenerRegistry::isListenerRegistered(server_port));

        // Stop progress thread before creating client connection
        registry_listener->stopProgressThread();

        // Use the SAME worker as the listener (like official UCXX example)
        auto shared_worker = registry_listener->getWorker();
        auto client_connection = std::make_unique<TestClient>(shared_worker, "127.0.0.1", server_port);
        REQUIRE(client_connection.get() != nullptr);
        //
        // IMPORTANT: Start progress thread again after creating client connection (following official UCXX pattern)
        registry_listener->startProgressThread();
        //
        // Wait for the registry listener to accept the connection
        CHECK(registry_listener->waitForConnection(5000));
        CHECK(registry_listener->isConnected());
        //
        // UCXConnection has been removed - use listener's tag methods directly
        //
        // IMPORTANT: Send small wireup messages first to let UCX identify capabilities
        // This is required before sending larger messages (following official UCXX pattern)
        std::vector<int> wireup_data = { 1, 2, 3 };
        std::vector<int> wireup_recv(wireup_data.size());

        std::string errorMsg;
        auto wireup_send_result = registry_listener->tagSend(
            wireup_data.data(), wireup_data.size() * sizeof(int), 0, errorMsg, std::optional<uint32_t>{ 5000 });
        auto wireup_recv_req = client_connection->tagReceive(
            wireup_recv.data(), wireup_recv.size() * sizeof(int), ucxx::Tag{ 0 }, ucxx::TagMaskFull);

        // Wait for wireup to complete
        auto start_time = std::chrono::steady_clock::now();
        auto timeout = std::chrono::seconds(5);
        while ((std::chrono::steady_clock::now() - start_time < timeout) && (!wireup_recv_req->isCompleted()))
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        CHECK(wireup_send_result == UcxSendResult::eSuccess);
        CHECK(wireup_recv_req->isCompleted());
        wireup_recv_req->checkError();
        //
        // Test bidirectional communication
        //
        // Scenario 1: Client sends data to server through registry listener
        std::vector<int> client_send_data = { 10, 20, 30, 40, 50 };
        std::vector<int> server_recv_data(client_send_data.size());
        ucxx::Tag test_tag1{ 1 };

        // Initiate send from client
        std::shared_ptr<ucxx::Request> client_send_request =
            client_connection->tagSend(client_send_data.data(), client_send_data.size() * sizeof(int), test_tag1);
        REQUIRE(client_send_request.get() != nullptr);

        // Initiate receive on server with timeout
        std::string recvErrorMsg;
        auto server_recv_result =
            registry_listener->tagReceive(server_recv_data.data(), server_recv_data.size() * sizeof(int), test_tag1,
                                          ucxx::TagMaskFull, recvErrorMsg, 5000);
        REQUIRE(server_recv_result == UcxReceiveResult::eSuccess);

        // Wait for client send to complete
        start_time = std::chrono::steady_clock::now();
        timeout = std::chrono::seconds(5);
        while ((std::chrono::steady_clock::now() - start_time < timeout) && (!client_send_request->isCompleted()))
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        CHECK(client_send_request->isCompleted());
        client_send_request->checkError();
        CHECK(server_recv_data == client_send_data);

        // Scenario 2: Server sends data to client through registry listener
        std::vector<int> server_send_data(100, 0);
        std::iota(server_send_data.begin(), server_send_data.end(), 100);
        std::vector<int> client_recv_data(server_send_data.size());
        ucxx::Tag test_tag2{ 2 };

        // Initiate send from server with timeout
        std::string sendErrorMsg;
        auto server_send_result =
            registry_listener->tagSend(server_send_data.data(), server_send_data.size() * sizeof(int), test_tag2,
                                       sendErrorMsg, std::optional<uint32_t>{ 5000 });
        REQUIRE(server_send_result == UcxSendResult::eSuccess);

        // Initiate receive on client
        std::shared_ptr<ucxx::Request> client_recv_request = client_connection->tagReceive(
            client_recv_data.data(), client_recv_data.size() * sizeof(int), test_tag2, ucxx::TagMaskFull);
        REQUIRE(client_recv_request.get() != nullptr);

        // Wait for client receive to complete
        start_time = std::chrono::steady_clock::now();

        while ((std::chrono::steady_clock::now() - start_time < timeout) && (!client_recv_request->isCompleted()))
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        CHECK(client_recv_request->isCompleted());
        REQUIRE_NOTHROW(client_recv_request->checkError());
        CHECK(client_recv_data == server_send_data);

        // Clean up - stop progress thread before destroying endpoints (following official UCXX shutdown pattern)
        shared_worker->stopProgressThread();
        client_connection.reset();
        registry_listener->shutdown();
    }

    TEST_CASE("UCXListenerRegistry: multi-buffer communication through registry")
    {
        // Clean up any existing listeners before starting
        UCXListenerRegistry::shutdown();

        // Create a listener through the registry using addListener for ephemeral port
        auto registry_listener = UCXListenerRegistry::addListener();
        REQUIRE(registry_listener.get() != nullptr);

        uint16_t server_port = registry_listener->getPort();
        CHECK(server_port != 0);

        // Stop progress thread before creating client connection
        registry_listener->stopProgressThread();

        // Use the SAME worker as the listener (like official UCXX example)
        auto shared_worker = registry_listener->getWorker();
        auto client_connection = std::make_unique<TestClient>(shared_worker, "127.0.0.1", server_port);

        // IMPORTANT: Start progress thread again after creating client connection (following official UCXX pattern)
        registry_listener->startProgressThread();

        // Wait for connection
        CHECK(registry_listener->waitForConnection(5000));

        // UCXConnection has been removed - use listener's tag methods directly
        // IMPORTANT: Send small wireup messages first to let UCX identify capabilities
        // This is required before sending larger messages (following official UCXX pattern)
        std::vector<int> wireup_data = { 1, 2, 3 };
        std::vector<int> wireup_recv(wireup_data.size());

        std::string wireupErrorMsg;
        auto wireup_send_result = registry_listener->tagSend(
            wireup_data.data(), wireup_data.size() * sizeof(int), 0, wireupErrorMsg, std::optional<uint32_t>{ 10000 });
        auto wireup_recv_req = client_connection->tagReceive(
            wireup_recv.data(), wireup_recv.size() * sizeof(int), ucxx::Tag{ 0 }, ucxx::TagMaskFull);

        // Wait for wireup to complete
        auto start_time = std::chrono::steady_clock::now();
        auto timeout = std::chrono::seconds(10);
        while ((std::chrono::steady_clock::now() - start_time < timeout) && (!wireup_recv_req->isCompleted()))
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        CHECK(wireup_send_result == UcxSendResult::eSuccess);
        CHECK(wireup_recv_req->isCompleted());
        wireup_recv_req->checkError();

        // Test multi-buffer send from client to server
        std::vector<int> buffer1 = { 1, 2, 3, 4, 5 };
        std::vector<float> buffer2 = { 1.1f, 2.2f, 3.3f };
        std::vector<char> buffer3 = { 'a', 'b', 'c', 'd' };

        std::vector<const void*> send_buffers = { buffer1.data(), buffer2.data(), buffer3.data() };
        std::vector<size_t> send_sizes = { buffer1.size() * sizeof(int), buffer2.size() * sizeof(float),
                                           buffer3.size() * sizeof(char) };
        std::vector<int> is_cuda = { 0, 0, 0 };
        ucxx::Tag multi_tag{ 3 };
        // Send multi-buffer from client
        std::shared_ptr<ucxx::Request> multi_send_request =
            client_connection->tagMultiSend(send_buffers, send_sizes, is_cuda, multi_tag);
        REQUIRE(multi_send_request.get() != nullptr);
        // Receive multi-buffer on server with timeout
        std::string multiRecvErrorMsg;
        auto multi_recv_result =
            registry_listener->tagMultiReceive(multi_tag, ucxx::TagMaskFull, multiRecvErrorMsg, 10000);
        REQUIRE(multi_recv_result == UcxReceiveResult::eSuccess);
        // Wait for client send to complete
        start_time = std::chrono::steady_clock::now();
        timeout = std::chrono::seconds(10);
        while ((std::chrono::steady_clock::now() - start_time < timeout) && (!multi_send_request->isCompleted()))
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        CHECK(multi_send_request->isCompleted());
        multi_send_request->checkError();
        // TODO(rhua): Verify the multi-buffer receive worked (basic completion check)
        // Note: Detailed buffer content verification would require accessing
        // the received buffers from the multi-recv request, which depends on
        // the specific UCX implementation details
        // Clean up - stop progress thread before destroying endpoints (following official UCXX shutdown pattern)
        shared_worker->stopProgressThread();
        client_connection.reset();
        registry_listener->shutdown();
    }

    TEST_CASE("UCXListenerRegistry: concurrent access and thread safety")
    {
        // Clean up any existing listeners before starting
        UCXListenerRegistry::shutdown();

        // Test concurrent creation of listeners
        std::vector<std::thread> threads;
        std::vector<std::shared_ptr<UCXListener>> listeners(5);
        std::vector<uint16_t> ports(5);

        // Create multiple listeners concurrently
        for (int i = 0; i < 5; ++i)
        {
            threads.emplace_back(
                [&listeners, &ports, i]()
                {
                    listeners[i] = UCXListenerRegistry::addListener();
                    if (listeners[i])
                    {
                        ports[i] = listeners[i]->getPort();
                    }
                });
        }

        // Wait for all threads to complete
        for (auto& thread : threads)
        {
            thread.join();
        }

        // Verify all listeners were created successfully
        for (int i = 0; i < 5; ++i)
        {
            REQUIRE(listeners[i].get() != nullptr);
            CHECK(ports[i] != 0);
            CHECK(UCXListenerRegistry::isListenerRegistered(ports[i]));
        }

        // Verify all ports are unique
        for (int i = 0; i < 5; ++i)
        {
            for (int j = i + 1; j < 5; ++j)
            {
                CHECK(ports[i] != ports[j]);
            }
        }

        // Test concurrent removal
        threads.clear();
        for (int i = 0; i < 3; ++i)
        {
            threads.emplace_back([&ports, i]() { UCXListenerRegistry::removeListener(ports[i]); });
        }

        for (auto& thread : threads)
        {
            thread.join();
        }

        // Verify removed listeners are no longer registered
        for (int i = 0; i < 3; ++i)
        {
            CHECK_FALSE(UCXListenerRegistry::isListenerRegistered(ports[i]));
        }

        // Verify remaining listeners are still registered
        for (int i = 3; i < 5; ++i)
        {
            CHECK(UCXListenerRegistry::isListenerRegistered(ports[i]));
        }

        // Clean up
        UCXListenerRegistry::shutdown();
    }

    TEST_CASE("UCXListenerRegistry: addListener API for ephemeral ports")
    {
        // Clean up any existing listeners before starting
        UCXListenerRegistry::shutdown();

        // Test that addListener creates listeners on ephemeral ports
        auto listener1 = UCXListenerRegistry::addListener();
        REQUIRE(listener1.get() != nullptr);

        uint16_t port1 = listener1->getPort();
        CHECK(port1 != 0);
        CHECK(UCXListenerRegistry::isListenerRegistered(port1));

        // Test that another addListener call gets a different port
        auto listener2 = UCXListenerRegistry::addListener();
        REQUIRE(listener2.get() != nullptr);

        uint16_t port2 = listener2->getPort();
        CHECK(port2 != 0);
        CHECK(port2 != port1);
        CHECK(listener2.get() != listener1.get());

        // Test that addListener can retrieve existing listeners by port
        auto same_listener1 = UCXListenerRegistry::addListener(port1);
        CHECK(same_listener1.get() == listener1.get());

        auto same_listener2 = UCXListenerRegistry::addListener(port2);
        CHECK(same_listener2.get() == listener2.get());

        // Test creating a listener on a specific port with addListener
        auto listener3 = UCXListenerRegistry::addListener(54321);
        REQUIRE(listener3.get() != nullptr);
        CHECK(listener3->getPort() == 54321);
        CHECK(UCXListenerRegistry::isListenerRegistered(54321));

        UCXListenerRegistry::shutdown();
    }

    TEST_CASE("UCXListenerRegistry: error handling and edge cases")
    {
        // Clean up any existing listeners before starting
        UCXListenerRegistry::shutdown();

        // Test removing non-existent listener (should not crash)
        UCXListenerRegistry::removeListener(12345);

        // Test checking non-existent listener
        CHECK_FALSE(UCXListenerRegistry::isListenerRegistered(12345));

        // Test multiple shutdowns (should not crash)
        UCXListenerRegistry::shutdown();
        UCXListenerRegistry::shutdown();

        // Test creating listener after shutdown
        auto listener = UCXListenerRegistry::addListener();
        REQUIRE(listener.get() != nullptr);
        uint16_t port = listener->getPort();
        CHECK(port != 0);
        CHECK(UCXListenerRegistry::isListenerRegistered(port));

        // Final cleanup
        UCXListenerRegistry::shutdown();
    }

    TEST_CASE("UCXListenerRegistry: tryRemoveListener conditional removal")
    {
        // Clean up any existing listeners before starting
        UCXListenerRegistry::shutdown();

        SUBCASE("returns false for non-existent listener")
        {
            bool result = UCXListenerRegistry::tryRemoveListener(12345);
            CHECK_FALSE(result);
        }

        SUBCASE("returns false when other references exist")
        {
            // Create a listener and hold a reference
            auto listener = UCXListenerRegistry::addListener();
            REQUIRE(listener.get() != nullptr);

            uint16_t port = listener->getPort();
            CHECK(UCXListenerRegistry::isListenerRegistered(port));

            // Try to remove while we still hold a reference - should fail
            bool result = UCXListenerRegistry::tryRemoveListener(port);
            CHECK_FALSE(result);

            // Listener should still be registered
            CHECK(UCXListenerRegistry::isListenerRegistered(port));

            // Clean up
            UCXListenerRegistry::shutdown();
        }

        SUBCASE("returns true and removes when only registry holds reference")
        {
            uint16_t port = 0;
            {
                // Create a listener
                auto listener = UCXListenerRegistry::addListener();
                REQUIRE(listener.get() != nullptr);
                port = listener->getPort();

                CHECK(UCXListenerRegistry::isListenerRegistered(port));

                // Release our reference (listener goes out of scope here)
            }

            // Now only the registry holds a reference
            // tryRemoveListener should succeed
            bool result = UCXListenerRegistry::tryRemoveListener(port);
            CHECK(result);

            // Listener should no longer be registered
            CHECK_FALSE(UCXListenerRegistry::isListenerRegistered(port));
        }

        SUBCASE("returns false with multiple node references, true after all released")
        {
            // Simulate multiple nodes sharing the same listener
            auto listener1 = UCXListenerRegistry::addListener();
            REQUIRE(listener1.get() != nullptr);

            uint16_t port = listener1->getPort();

            // Second node gets the same listener
            auto listener2 = UCXListenerRegistry::addListener(port);
            REQUIRE(listener2.get() == listener1.get());

            // Third node gets the same listener
            auto listener3 = UCXListenerRegistry::addListener(port);
            REQUIRE(listener3.get() == listener1.get());

            // First node resets and tries to remove - should fail (2 others still hold refs)
            listener1.reset();
            bool result1 = UCXListenerRegistry::tryRemoveListener(port);
            CHECK_FALSE(result1);
            CHECK(UCXListenerRegistry::isListenerRegistered(port));

            // Second node resets and tries to remove - should fail (1 other still holds ref)
            listener2.reset();
            bool result2 = UCXListenerRegistry::tryRemoveListener(port);
            CHECK_FALSE(result2);
            CHECK(UCXListenerRegistry::isListenerRegistered(port));

            // Third node resets and tries to remove - should succeed (no other refs)
            listener3.reset();
            bool result3 = UCXListenerRegistry::tryRemoveListener(port);
            CHECK(result3);
            CHECK_FALSE(UCXListenerRegistry::isListenerRegistered(port));
        }

        // Final cleanup
        UCXListenerRegistry::shutdown();
    }
}
