// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

#include "IExecutionProvider.h"
#include "ExecutionProvider.h"
#include "core/providers/dml/OperatorAuthorHelper/MLOperatorAuthorHelper.h"
#include "FusedGraphKernel.h"
#include "GraphDescBuilder.h"
#include "core/graph/indexed_sub_graph.h"
#include "core/framework/compute_capability.h"
#include <wil/wrl.h>
#include <dxgi1_6.h>
#include "GraphPartitioner.h"

//#define PRINT_PARTITON_INFO
  
using namespace winrt::Windows::AI::MachineLearning::implementation;

namespace Dml
{
    GraphPartition* GraphPartition::GetRootMergedPartition()
    {
        return m_mergedPartition ? m_mergedPartition->GetRootMergedPartition() : this;
    }

    std::vector<onnxruntime::NodeIndex>& GraphPartition::GetNodeIndices()
    {
        assert(this == GetRootMergedPartition());
        return m_nodeIndices;
    }

    std::set<std::string>& GraphPartition::GetInputs()
    {
        assert(this == GetRootMergedPartition());
        return m_inputs;
    }

    std::set<std::string>& GraphPartition::GetOutputs()
    {
        assert(this == GetRootMergedPartition());
        return m_outputs;
    }

    bool GraphPartition::IsFinalized()
    {
        assert(this == GetRootMergedPartition());
        return m_finalized;
    }

    void GraphPartition::SetFinalized()
    {
        m_finalized = true;
    }

    bool GraphPartition::IsDmlPartition()
    {
        assert(this == GetRootMergedPartition());
        return m_isDmlPartition;
    }

    bool GraphPartition::IsDmlGraphPartition()
    {
        assert(this == GetRootMergedPartition());
        return m_isDmlGraphPartition;
    }

    void GraphPartition::SetIsDmlPartition(bool isDmlPartition)
    {
        assert(this == GetRootMergedPartition());
        m_isDmlPartition = isDmlPartition;
    }

    void GraphPartition::SetIsDmlGraphPartition(bool isDmlGraphPartition)
    {
        assert(this == GetRootMergedPartition());
        m_isDmlGraphPartition = isDmlGraphPartition;
    }

    void GraphPartition::AddNodeIndex(onnxruntime::NodeIndex index)
    {
        assert(!IsFinalized());
        assert(std::find(m_nodeIndices.begin(), m_nodeIndices.end(), index) == m_nodeIndices.end());

        m_nodeIndices.push_back(index);
    }
     
    void GraphPartition::AddInput(const std::string& name)
    {
        assert(!IsFinalized());
        assert(this == GetRootMergedPartition());
        m_inputs.insert(name);
    }

    void GraphPartition::AddOutput(const std::string& name)
    {
        assert(this == GetRootMergedPartition());
        m_outputs.insert(name);
    }

    void GraphPartition::Merge(gsl::span<GraphPartition*> partitionsToMerge)
    {
        assert(this == GetRootMergedPartition());

        for (GraphPartition* partitionToMerge : partitionsToMerge)
        {
            if (partitionToMerge == this)
            {
                continue;
            }

            assert(!partitionToMerge->IsFinalized());
            assert(partitionToMerge->IsDmlPartition() == IsDmlPartition());
            assert(partitionToMerge->IsDmlGraphPartition() == IsDmlGraphPartition());

            partitionToMerge->m_mergedPartition = this;
                
            m_nodeIndices.insert(m_nodeIndices.begin(), partitionToMerge->m_nodeIndices.begin(), partitionToMerge->m_nodeIndices.end());
            m_inputs.insert(partitionToMerge->m_inputs.begin(), partitionToMerge->m_inputs.end());
            m_outputs.insert(partitionToMerge->m_outputs.begin(), partitionToMerge->m_outputs.end());
        }
    }

    // Adds the outputs of a node to the specified partition
    void AddNodeOutputsToPartitionMap(
        const onnxruntime::Node& node, 
        GraphPartition* partition,
        std::unordered_map<std::string, GraphPartition*>& nodeNameToPartitionMap
    )
    {
        for (uint32_t i = 0; i < node.OutputDefs().size(); ++i) 
        {
            const auto* arg = node.OutputDefs()[i];
            if (arg->Exists())
            {
                nodeNameToPartitionMap[arg->Name()] = partition;
            }
        }
    };

    bool NodeArgSupportedInGraph(const onnxruntime::NodeArg* arg, bool requiresFloatFormats)
    {            
        if (arg->Exists())
        {
            const onnx::TypeProto* typeProto = arg->TypeAsProto();
            if (typeProto->value_case() == onnx::TypeProto::kTensorType)
            {
                const onnx::TypeProto_Tensor tensorType = typeProto->tensor_type();
                if (tensorType.has_elem_type())
                {
                    // TODO: Remove this by handling zeroing on the output of fused graph nodes and handling of non-float 
                    // types in DML's identity operator, which is used for strided copies.
                    if (ToMLTensorDataType(static_cast<onnx::TensorProto_DataType>(tensorType.elem_type())) == MLOperatorTensorDataType::UInt64 ||
                        ToMLTensorDataType(static_cast<onnx::TensorProto_DataType>(tensorType.elem_type())) == MLOperatorTensorDataType::Int64)
                    {
                        return false;
                    }

                    if (requiresFloatFormats)
                    {
                        if (ToMLTensorDataType(static_cast<onnx::TensorProto_DataType>(tensorType.elem_type())) != MLOperatorTensorDataType::Float &&
                            ToMLTensorDataType(static_cast<onnx::TensorProto_DataType>(tensorType.elem_type())) != MLOperatorTensorDataType::Float16)
                        {
                            return false;
                        }
                    }
                }
            }
        }

        return true;
    }

    bool NodeTensorTypesSupportedInGraph(const onnxruntime::Node& node, const InternalRegistrationInfo& registration)
    {
        for (size_t i = 0; i < node.InputDefs().size(); ++i)
        {
            bool isConstantCpuInput = std::find(registration.requiredConstantCpuInputs.begin(), registration.requiredConstantCpuInputs.end(), i) !=
                  registration.requiredConstantCpuInputs.end();

            if (!isConstantCpuInput && !NodeArgSupportedInGraph(node.InputDefs()[i], registration.graphNodeFactoryRegistration->requiresFloatFormatsExceptConstInputs))
            {
                return false;
            }
        }

        for (auto arg : node.OutputDefs())
        {
            if (!NodeArgSupportedInGraph(arg, registration.graphNodeFactoryRegistration->requiresFloatFormatsExceptConstInputs))
            {
                return false;
            }
        }

        return true;
    }

    bool TryGetTensorDataType(
        const onnxruntime::NodeArg& nodeArg,
        _Out_ MLOperatorTensorDataType* onnxElementType
    )
    {
        *onnxElementType = MLOperatorTensorDataType::Undefined;

        const ::onnx::TypeProto* typeProto = nodeArg.TypeAsProto();
        if (typeProto != nullptr && typeProto->has_tensor_type())
        {
            const ::onnx::TypeProto_Tensor& tensorTypeProto = typeProto->tensor_type();
            if (tensorTypeProto.has_elem_type())
            {
                *onnxElementType = static_cast<MLOperatorTensorDataType>(tensorTypeProto.elem_type());
                return true;
            }
        }

        return false;
    }

    bool DoesNodeContainSupportedDataTypes(
        const onnxruntime::Node& node,
        bool allow64BitInputThroughStrides,
        _In_opt_ const std::unordered_map<std::string, GraphPartition*>* nodeNameToPartitionMap, // Only used when allow64BitInputThroughStrides is true        
        _In_opt_ const InternalRegistrationInfo* regInfo,
        uint32_t supportedDeviceDataTypeMask // Each bit corresponds to each DML_TENSOR_DATA_TYPE.
        )
    {
        THROW_HR_IF(E_INVALIDARG, allow64BitInputThroughStrides && !nodeNameToPartitionMap);

        // Assume data types are supported until proven otherwise.
        bool nodeContainsSupportedDataTypes = true;

        // Callback to check each node's data type.
        std::function<void(const onnxruntime::NodeArg& nodeArg, bool isInput)> nodeCallback = [&](const onnxruntime::NodeArg& nodeArg, bool isInput) -> void
        {
            // Get the tensor element data type for this node, comparing against what the device actually supports.
            // Use the enumeration from the proto instead of nodeArg.Type() which returns a string.
            MLOperatorTensorDataType onnxElementType;
            if (TryGetTensorDataType(nodeArg, &onnxElementType))
            {
                DML_TENSOR_DATA_TYPE dmlElementType = GetDmlDataTypeFromMlDataTypeNoThrow(onnxElementType);
                if (dmlElementType != DML_TENSOR_DATA_TYPE_UNKNOWN)
                {
                    if (((1 << dmlElementType) & supportedDeviceDataTypeMask) == 0)
                    {
                        nodeContainsSupportedDataTypes = false;
                    }
                }
            }
        };

        // Check whether the node uses any data types which are unsupported by the device.
        node.ForEachDef(nodeCallback);

        // DML kernels supporting int64 and uint64 are expected to not *introduce* values out of range, which allows
        // the temporary trick using strides to emulate 64 bit tensors to work.  If the source is a CPU operator, 
        // graph input or initializer, it's not safe to assume the input can be represented with 32 bits.
        if (regInfo)
        {
            for (uint32_t i = 0; i < node.InputDefs().size(); ++i) 
            {
                const auto* arg = node.InputDefs()[i];
                MLOperatorTensorDataType onnxElementType;
                if (arg->Exists() && TryGetTensorDataType(*arg, &onnxElementType))
                {
                    if (((onnxElementType == MLOperatorTensorDataType::UInt64) || (onnxElementType == MLOperatorTensorDataType::Int64)))
                    {
                        // Look up the input partition.  If it's a graph input or initializer it will be missing
                        // from the map.  In this case or if the input comes from a CPU partition, it might be 
                        // out of range.
                        const std::string& argName = arg->Name(); 
                        // Check if the operator handles the input on the CPU as a constant input
                        bool isConstantCpuInput = std::find(regInfo->requiredConstantCpuInputs.begin(), regInfo->requiredConstantCpuInputs.end(), i) !=
                                regInfo->requiredConstantCpuInputs.end();

                        if (!isConstantCpuInput)
                        {
                            if (!allow64BitInputThroughStrides)
                            {
                                nodeContainsSupportedDataTypes = false;
                                break;
                            }

                            auto partitionIter = nodeNameToPartitionMap->find(argName);
                            if (partitionIter == nodeNameToPartitionMap->end() || !partitionIter->second->IsDmlPartition())
                            {
                                nodeContainsSupportedDataTypes = false;
                                break;
                            }
                        }
                    }
                }
            }
        }

        return nodeContainsSupportedDataTypes;
    }

    bool IsNodeSupportedByDml(
        const onnxruntime::Node& node,    
        const onnxruntime::KernelRegistry& registry,
        uint32_t supportedDeviceDataTypeMask, // Each bit corresponds to each DML_TENSOR_DATA_TYPE.
        const InternalRegistrationInfoMap& internalRegInfoMap,

        bool allow64BitInputThroughStrides,
        _In_opt_ const std::unordered_map<std::string, GraphPartition*>* nodeNameToPartitionMap
    )
    {
        THROW_HR_IF(E_INVALIDARG, allow64BitInputThroughStrides && !nodeNameToPartitionMap);

        const onnxruntime::KernelCreateInfo* createInfo = registry.TryFindKernel(node, onnxruntime::kDmlExecutionProvider);
        if (!createInfo)
        {
            return false;
        }

        auto regInfoIter = internalRegInfoMap.find(createInfo->kernel_def.get());
        std::shared_ptr<InternalRegistrationInfo> internalRegInfo;
        if (regInfoIter != internalRegInfoMap.end())
        {
            internalRegInfo = regInfoIter->second;
            if (internalRegInfo->supportQuery && !internalRegInfo->supportQuery(node))
            {
                return false;
            }
        }

        // Check whether the node uses any data types which are unsupported by the device.
        if (!DoesNodeContainSupportedDataTypes(node, allow64BitInputThroughStrides, nodeNameToPartitionMap, internalRegInfo.get(), supportedDeviceDataTypeMask))
        {
            return false;
        }

        return true;
    }


    // Gets properties of the registration for a node
    void GetRegistrationProperties(
        const onnxruntime::GraphViewer& graph,
        const onnxruntime::Node& node,    
        const std::vector<const onnxruntime::KernelRegistry*>& dmlRegistries,
        uint32_t supportedDeviceDataTypeMask, // Each bit corresponds to each DML_TENSOR_DATA_TYPE.
        const InternalRegistrationInfoMap& internalRegInfoMap,
        _In_opt_ const std::unordered_map<std::string, GraphPartition*>* nodeNameToPartitionMap,
        _Inout_ std::unordered_map<const onnxruntime::Node*, GraphNodeProperties>& dmlNodePropertyMap,
        _Inout_ std::unordered_set<std::string>& requiredInitializerMap,
        _Out_ bool* isDmlNode,
        _Out_ bool* isDmlGraphNode
        )
    {
        *isDmlNode = false;
        *isDmlGraphNode = false;

        // Find the highest priority DML registry supporting this node, and get its highest-priority
        // registration.  Determine if that registration supports usage as a graph node.
        for (auto registry : dmlRegistries) 
        {
            bool allow64BitInputThroughStrides = true;
            if (IsNodeSupportedByDml(node, *registry, supportedDeviceDataTypeMask, internalRegInfoMap, allow64BitInputThroughStrides, nodeNameToPartitionMap))
            {
                *isDmlNode = true;

                // Get the kernel creation info for the registration, and check if it carries the property
                // set during registration of kernels that support DML graph node usage.
                auto& graphNodeProperty = dmlNodePropertyMap.insert(std::make_pair(&node, GraphNodeProperties()));

                // Ensure that shape information is known statically for the inputs and outputs of the node,
                // which is required for MLGraph compilation.
                const onnxruntime::KernelCreateInfo* createInfo = registry->TryFindKernel(node, onnxruntime::kDmlExecutionProvider);

                auto regInfoIter = internalRegInfoMap.find(createInfo->kernel_def.get());
                if (regInfoIter != internalRegInfoMap.end())
                {
                    auto internalRegInfo = regInfoIter->second;

                    if (internalRegInfo && internalRegInfo->graphNodeFactoryRegistration && 
                        NodeTensorTypesSupportedInGraph(node, *internalRegInfo))
                    {
                        bool requiredCpuInputsConstant = true;
                        for (uint32_t inputIndex : internalRegInfo->requiredConstantCpuInputs)
                        {
                            if (inputIndex >= node.InputDefs().size() || !node.InputDefs()[inputIndex]->Exists())
                            {
                                continue;
                            }

                            const onnx::TensorProto* tensor = nullptr;
                            const std::string& inputName = node.InputDefs()[inputIndex]->Name();

                            if (!graph.GetInitializedTensor(inputName, tensor))
                            {
                                requiredCpuInputsConstant = false;
                                break;
                            }

                            requiredInitializerMap.insert(inputName);
                        }

                        std::optional<uint32_t> requiredInputCount = internalRegInfo->graphNodeFactoryRegistration->requiredInputCount;
                        if (requiredCpuInputsConstant &&
                            TryGetStaticInputShapes( node, graphNodeProperty.first->second.inputShapes) &&
                            !ContainsEmptyDimensions(graphNodeProperty.first->second.inputShapes) &&
                            TryGetStaticOutputShapes(node, graphNodeProperty.first->second.outputShapes) &&
                            !ContainsEmptyDimensions(graphNodeProperty.first->second.outputShapes) &&
                            (requiredInputCount == std::nullopt || *requiredInputCount == node.InputDefs().size()))
                        {
                            *isDmlGraphNode = true;
                             graphNodeProperty.first->second.internalRegInfo = internalRegInfo;
                        }
                    }
                }

                break;
            }
        }
    }

    // Creates a partition for a node which is not a DML graph node, and finalizes partitions
    // which are inputs of the new partition.
    std::unique_ptr<GraphPartition> CreateNonGraphNodePartitionAndFinalizeInputs(
        const onnxruntime::Node& node, 
        bool isDmlNode,
        std::unordered_map<std::string, GraphPartition*>& nodeNameToPartitionMap
    )
    {
        std::unique_ptr<GraphPartition> partition = std::make_unique<GraphPartition>();
        partition->SetIsDmlGraphPartition(false);
        partition->SetIsDmlPartition(isDmlNode);
        partition->AddNodeIndex(node.Index());

        for (uint32_t i = 0; i < node.InputDefs().size(); ++i) 
        {
            const auto* arg = node.InputDefs()[i];
            if (arg->Exists())
            {
                const std::string& argName = arg->Name();
                        
                if (nodeNameToPartitionMap.find(argName) != nodeNameToPartitionMap.end())
                {
                    // Finalize the partition which contains an input to a non-DML-graph partition.
                    // The connections from that partition to other partitions, such as this one, 
                    // must become outputs of that partition.  As subsequent downstream nodes of 
                    // the finalized partition are visited, other outputs will subsequently be 
                    // added to the partition, too.
                    GraphPartition* inputPartition = nodeNameToPartitionMap[argName]->GetRootMergedPartition();
                    inputPartition->SetFinalized();
                    inputPartition->AddOutput(argName);
                }

                partition->AddInput(argName);
            }
        }

        partition->SetFinalized();
        AddNodeOutputsToPartitionMap(node, partition.get(), nodeNameToPartitionMap);

        return partition;
    }

    // Get the partitions which are inputs to the specified node and which are not finalized.
    std::vector<GraphPartition*> GetNonFinalizedInputPartitions(
        const onnxruntime::Node& node, 
        std::unordered_map<std::string, GraphPartition*>& nodeNameToPartitionMap
    )
    {
        std::vector<GraphPartition*> inputNonFinalPartitions;

        for (uint32_t i = 0; i < node.InputDefs().size(); ++i) 
        {
            const auto* arg = node.InputDefs()[i];
            if (arg->Exists())
            {
                const std::string& argName = arg->Name();

                if (nodeNameToPartitionMap.find(argName) == nodeNameToPartitionMap.end())
                {
                    // Must be source node
                    continue;
                }

                GraphPartition* inputPartition = nodeNameToPartitionMap[argName]->GetRootMergedPartition();

                if (!inputPartition->IsFinalized())
                {
                    inputNonFinalPartitions.push_back(inputPartition);
                }
            }
        }

        return inputNonFinalPartitions;
    }
   
    // Add graph outputs of the new node to a partition.
    void AddGraphOutputsFromNodeToPartition(
        const onnxruntime::Node& node, 
        const std::set<std::string>& graphOutputs,
        GraphPartition* partition
    )
    {
        for (uint32_t i = 0; i < node.OutputDefs().size(); ++i) 
        {
            const auto* arg = node.OutputDefs()[i];
            if (arg->Exists())
            {
                if (graphOutputs.find(arg->Name()) != graphOutputs.end())
                {
                    partition->AddOutput(arg->Name());
                }
            }
        }
    }

    std::unique_ptr<GraphPartition> CreateNewPartitionWithFinalizedInputPartitions(
        const onnxruntime::Node& node, 
        const std::set<std::string>& graphOutputs,
        std::unordered_map<std::string, GraphPartition*>& nodeNameToPartitionMap
    )
    {
        std::unique_ptr<GraphPartition> partition = std::make_unique<GraphPartition>();
        partition->SetIsDmlGraphPartition(true);
        partition->SetIsDmlPartition(true);
        partition->AddNodeIndex(node.Index());

        // Inputs of the partition are added when partitions are created and extended when
        // nodes are added with inputs which are not inside the partition
        for (uint32_t i = 0; i < node.InputDefs().size(); ++i) 
        {
            const auto* arg = node.InputDefs()[i];
            if (arg->Exists())
            {
                partition->AddInput(arg->Name());

                auto& inputPartition = nodeNameToPartitionMap.find(arg->Name());
                if (inputPartition != nodeNameToPartitionMap.end())
                {
                    inputPartition->second->GetRootMergedPartition()->AddOutput(arg->Name());
                }
            }
        }

        // Outputs of the partition are initially set to node outputs which are also
        // graph outputs.  They are extended when adding other node with the graph
        // outputs from those nodes.  They are also extended when a partition
        // consumes an input from the current partition.
        AddGraphOutputsFromNodeToPartition(node, graphOutputs, partition.get());

        AddNodeOutputsToPartitionMap(node, partition.get(), nodeNameToPartitionMap);

        return partition;
    }
    
    std::unique_ptr<onnxruntime::ComputeCapability> ComputationCapacityFromPartition(
        GraphPartition* partition, 
        uint32_t partitionIndex, 
        const onnxruntime::GraphViewer& graph, 
        std::unordered_map<const onnxruntime::Node*, GraphNodeProperties>&& graphNodePropertyMap,
        onnxruntime::KernelRegistry* registryForPartitionKernels,
        const std::string& partitionKernelPrefix,
        std::shared_ptr<std::unordered_map<std::string, onnx::TensorProto>> transferredInitializerMap)
    {
        std::unique_ptr<onnxruntime::IndexedSubGraph> subGraph = std::make_unique<onnxruntime::IndexedSubGraph>();

        if (partition->IsDmlGraphPartition())
        {
            assert(partition->IsDmlGraphPartition());
         
            // Create a definition for the node.  The name must be unique.
            auto def = std::make_unique<onnxruntime::IndexedSubGraph::MetaDef>();
            def->name = std::string("DmlFusedNode_") + partitionKernelPrefix + std::to_string(partitionIndex);
            def->domain = "DmlFusedNodeDomain";
            def->since_version = 1;
            def->inputs.insert(def->inputs.begin(), partition->GetInputs().begin(), partition->GetInputs().end());
            def->outputs.insert(def->outputs.begin(), partition->GetOutputs().begin(), partition->GetOutputs().end());

            // Populate properties which will be passed to OpKernel for this graph via the function below
            std::unordered_map<std::string, GraphNodeProperties> partitionNodePropsMap;
            for (auto nodeIndex : partition->GetNodeIndices())
            {
                const onnxruntime::Node* node = graph.GetNode(nodeIndex);
                    
#ifdef PRINT_PARTITON_INFO
                printf("Partition %u\t%s\n", partitionIndex, GraphDescBuilder::GetUniqueNodeName(*node).c_str());
#endif
                partitionNodePropsMap.insert(std::make_pair(
                    GraphDescBuilder::GetUniqueNodeName(*node), std::move(graphNodePropertyMap[node])));
            }
            
#ifdef PRINT_PARTITON_INFO
            printf("\n");
#endif

            auto fused_kernel_func = [partitionNodePropsMap, transferredInitializerMap](const onnxruntime::OpKernelInfo& info) mutable ->onnxruntime::OpKernel*
            {
                return CreateFusedGraphKernel(info, partitionNodePropsMap, *transferredInitializerMap);
            };

            // build the kernel definition on the fly, and register it to the fused_kernel_regisitry.
            onnxruntime::KernelDefBuilder builder;

            builder.SetName(def->name)
                .SetDomain(def->domain)
                .SinceVersion(def->since_version)
                .Provider(onnxruntime::kDmlExecutionProvider);

            registryForPartitionKernels->Register(builder, fused_kernel_func);
            
            subGraph->SetMetaDef(std::move(def));
        }

        subGraph->nodes = std::move(partition->GetNodeIndices());

        return std::make_unique<onnxruntime::ComputeCapability>(std::move(subGraph));
    }

    // Whether any operator in the model contains a subgraph.  This is true
    // if the graph being partitioned is itself within a subgraph, or contains
    // an operator with a subgraph.
    bool ModelUsesSubgraph(const onnxruntime::GraphViewer& graph)
    {
        if (graph.IsSubgraph())
        {
            return true;
        }

        const std::vector<onnxruntime::NodeIndex>& toplogicalOrder = graph.GetNodesInTopologicalOrder();

        for (size_t nodeIndex : toplogicalOrder) 
        {
            const onnxruntime::Node& node = *graph.GetNode(nodeIndex);
            if (node.ContainsSubgraph())
            {
                return true;
            }
        }

        return false;
    }

    // 
    // A simple graph partitioning algorithm is used:
    //
    // - If a node has any input which is already in a graph, and that graph is not finalized,
    //   then the node and all such input graphs are merged.
    //
    // - Once a node has an output which cannot be merged with its graph, its graph is marked 
    //   as final, which disallows its future extensions.  This ensures that no indirect 
    //   downstream dependencies of the external output node are later merged.
    //
    std::vector<std::unique_ptr<GraphPartition>>
    BuildPartitions(
        const onnxruntime::GraphViewer& graph,
        const InternalRegistrationInfoMap& internalRegInfoMap,
        const std::vector<const onnxruntime::KernelRegistry*>& registries,
        uint32_t supportedDeviceDataTypeMask, // Each bit corresponds to each DML_TENSOR_DATA_TYPE.
        std::unordered_map<const onnxruntime::Node*, GraphNodeProperties>& graphNodePropertyMap,
        std::unordered_set<std::string>& requiredInitializerMap,
        std::function<void(const onnxruntime::Node&)> onNodeUnsupportedInGraph)
    {
        // Nodes are uniquely identified by the name of their first output argument
        std::vector<std::unique_ptr<GraphPartition>> partitions;
        std::unordered_map<std::string, GraphPartition*> nodeNameToPartitionMap;

        // Get the list of node indices in toplogical order, so nodes are visited before.
        // downstream nodes consuming them.
        const std::vector<onnxruntime::NodeIndex>& toplogicalOrder = graph.GetNodesInTopologicalOrder();

        // Construct sets with graph inputs and outputs for fast lookup later.
        std::set<std::string> graphInputs;
        std::set<std::string> graphOutputs;

        for (const auto* arg : graph.GetInputsIncludingInitializers())
        {
            graphInputs.insert(arg->Name());
        }        
        
        // If a model contains an intializer which is not also a graph input, it will not be returned
        // by GetInputsIncludingInitializers above.  Such models would be invalid, however they loaded
        // in RS5.  For compatibility, this ensures that such models continue to load.  This is 
        // verified by an ONNX conformance test for Add.
        for (const auto& arg : graph.GetAllInitializedTensors())
        {
            // This adds the initializer to the input set if it didn't already exist.
            graphInputs.insert(arg.first);
        }
        
        for (const auto* arg : graph.GetOutputs())
        {
            graphOutputs.insert(arg->Name());
        }
        
        // Check whether this graph is a subgraph, or contains any node with a subgraph. 
        bool modelUsesSubgraph = ModelUsesSubgraph(graph);

        // Build up partitions while traversing the graph.
        for (size_t nodeIndex : toplogicalOrder) 
        {
            const onnxruntime::Node& node = *graph.GetNode(nodeIndex);

            // Whether the node is implemented through DML.
            bool isDmlNode = false;

            // Whether the node is implemented through DML and as a graph node, meaning it
            // can generate DML operations through a private interface for use as an MLGraph node.
            bool isDmlGraphNode = false;
            
            // Get the registration properties above and populate nodeNameToPartitionMap.
            GetRegistrationProperties(
                graph,
                node,
                registries,
                supportedDeviceDataTypeMask,
                internalRegInfoMap,
                &nodeNameToPartitionMap,
                graphNodePropertyMap,
                requiredInitializerMap,
                /*out*/ &isDmlNode,
                /*out*/ &isDmlGraphNode
            );

            // Add a unique partition if graph node usage is not supported.
            //
            // Partitioning is disabled in models with subgraphs to work around issues with implicit inputs.  
            // The partitioning algorithm does not currently consider such inputs.  Transfering shared initializers 
            // for partitions could also cause problems.  Note, operators with subgraphs are currently not efficient 
            // anyhow due to CPU/GPU copies.
            if (modelUsesSubgraph || !isDmlGraphNode)
            {
                if (onNodeUnsupportedInGraph)
                {
                    onNodeUnsupportedInGraph(node);
                }

                partitions.push_back(CreateNonGraphNodePartitionAndFinalizeInputs(node, isDmlNode, nodeNameToPartitionMap));
                continue;
            }
            
            std::vector<GraphPartition*> inputNonFinalPartitions = GetNonFinalizedInputPartitions(node, nodeNameToPartitionMap);
                
            if (inputNonFinalPartitions.empty())
            {
                partitions.push_back(CreateNewPartitionWithFinalizedInputPartitions(node, graphOutputs, nodeNameToPartitionMap));
            }
            else
            {
                // Arbitrarily pick the first non-final partition found among the inputs, and add this node
                // and its output arguments to that partition.
                GraphPartition* firstNonFinalInputPartition = inputNonFinalPartitions[0]->GetRootMergedPartition();
                firstNonFinalInputPartition->AddNodeIndex(node.Index());
                AddNodeOutputsToPartitionMap(node, firstNonFinalInputPartition, nodeNameToPartitionMap);

                // Add inputs for the new node which span partitions
                for (uint32_t i = 0; i < node.InputDefs().size(); ++i) 
                {
                    const auto* arg = node.InputDefs()[i];
                    if (arg->Exists())
                    {
                        auto& inputPartition = nodeNameToPartitionMap.find(arg->Name());

                        // Add the input of the current node into the partition which the node will be merged into.
                        // Skip this if the input is already merged into the same partition or is not finalized,
                        // and so will be subsequently merged below.
                        if (inputPartition != nodeNameToPartitionMap.end() && 
                            inputPartition->second->GetRootMergedPartition() != firstNonFinalInputPartition &&
                            inputPartition->second->GetRootMergedPartition()->IsFinalized())
                        {
                            // Add this input of the current node as an output of the final partition to which 
                            // it belongs.  
                            inputPartition->second->GetRootMergedPartition()->AddOutput(arg->Name());
                            firstNonFinalInputPartition->AddInput(arg->Name());
                        }
                        
                        if (graphInputs.find(arg->Name()) != graphInputs.end())
                        { 
                            firstNonFinalInputPartition->AddInput(arg->Name());
                        }
                    }
                }

                // Add graph outputs of the new node
                AddGraphOutputsFromNodeToPartition(node, graphOutputs, firstNonFinalInputPartition);

                // Merge each other non-finalized input partition into the first one
                if (inputNonFinalPartitions.size() > 1)
                {
                    firstNonFinalInputPartition->Merge(gsl::span<GraphPartition*>(&inputNonFinalPartitions[1], inputNonFinalPartitions.size() - 1));
                } 
            }
        }

        return partitions;
    }

    std::unordered_map<const onnx::TensorProto*, std::vector<uint32_t>>
    GetInitializerToPartitionMap(
        const onnxruntime::GraphViewer& graph,
        gsl::span<std::unique_ptr<GraphPartition>> partitions
    )
    {
        std::unordered_map<const onnx::TensorProto*, std::vector<uint32_t>> initializerPartitionMap;

        for (uint32_t partitionIndex = 0; partitionIndex < gsl::narrow_cast<uint32_t>(partitions.size()); ++partitionIndex)
        {
            auto& partition = partitions[partitionIndex];

            // Skip partitions which have been merged into other partitions
            if (partition->GetRootMergedPartition() != partition.get())
            {
                continue;
            }

            std::unordered_map<std::string, onnx::TensorProto> transferredInitializerMap;

            for (const std::string& input : partition->GetInputs())
            {
                const onnx::TensorProto* tensor = nullptr;
                if (graph.GetInitializedTensor(input, tensor))
                {
                    initializerPartitionMap[tensor].push_back(partitionIndex);
                }
            }
        }

        return initializerPartitionMap;
    }

    std::vector<std::unique_ptr<onnxruntime::ComputeCapability>>
    PartitionGraph(
        const onnxruntime::GraphViewer& graph,
        const InternalRegistrationInfoMap& internalRegInfoMap,
        const std::vector<const onnxruntime::KernelRegistry*>& registries,
        uint32_t supportedDeviceDataTypeMask, // Each bit corresponds to each DML_TENSOR_DATA_TYPE.
        onnxruntime::KernelRegistry* registryForPartitionKernels,
        const std::string& partitionKernelPrefix
        )
    {
        std::vector<std::unique_ptr<onnxruntime::ComputeCapability>> result;

        // Initializers needed by any graph partition
        std::unordered_set<std::string> requiredInitializerMap;

        std::unordered_map<const onnxruntime::Node*, GraphNodeProperties> graphNodePropertyMap;
        std::vector<std::unique_ptr<GraphPartition>> partitions = BuildPartitions(
            graph,
            internalRegInfoMap, 
            registries,
            supportedDeviceDataTypeMask,
            graphNodePropertyMap, 
            requiredInitializerMap);

        // Create a map between each initialized tensor and the partition(s) it is part of.
        auto initializerPartitionMap = GetInitializerToPartitionMap(graph, partitions);

        for (uint32_t partitionIndex = 0; partitionIndex < partitions.size(); ++partitionIndex)
        {
            auto& partition = partitions[partitionIndex];

            if (partition->GetRootMergedPartition() != partition.get() ||
                !partition->IsDmlPartition())
            {
                continue;
            }

            // Create a map which will store by name each initializer which should be transferred to the 
            // partition.  This prevents OnnxRuntime from allocating GPU resources and uploading those initializers,
            // so the partiton's kernel can do so.  In the process, it will pre-process weights while consuming a CPU
            // backed resource, avoiding an extra set of GPU resources in memory.
            // A shared pointer is used so the functor and contained initializer captures can be cheaply copied within ORT.
            auto transferredInitializerMap = std::make_shared<std::unordered_map<std::string, onnx::TensorProto>>();

            for (const auto& input : partition->GetInputs())
            {
                if (partition->IsDmlGraphPartition())
                {
                    const onnx::TensorProto* tensor = nullptr;
                    if (graph.GetInitializedTensor(input, tensor))
                    {
                        // It's only safe to transfer tensors which are used by this partition alone.
                        auto iter = initializerPartitionMap.find(tensor);
                        assert(iter != initializerPartitionMap.end());
                        if (iter->second.size() > 1)
                        {
                            bool inputConstant = false;
                            if (requiredInitializerMap.find(input) != requiredInitializerMap.end())
                            {
                                // The kernel relies on this input to be initialized, and it should be small enough to copy
                                // cheaply. FusedGraphKernel only handles constant CPU inputs through transferred initializers,
                                // rather than ORT, to avoid mismatches in policy or implementation causing failures.
                                (*transferredInitializerMap)[input] = const_cast<onnx::TensorProto&>(*tensor);
                            }

                            continue;
                        }

                        // Transfer the initializer
                        auto& graphTensor = const_cast<onnx::TensorProto&>(*tensor);

                        onnx::TensorProto partitionTensor;
                        graphTensor.Swap(&partitionTensor);
                        (*transferredInitializerMap)[input] = std::move(partitionTensor);
                
                        const_cast<onnxruntime::InitializedTensorSet&>(graph.GetAllInitializedTensors()).erase(graph.GetAllInitializedTensors().find(input));
                    }
                }
            }

            result.push_back(ComputationCapacityFromPartition(
                partition.get(), 
                partitionIndex, 
                graph, 
                std::move(graphNodePropertyMap),
                registryForPartitionKernels,
                partitionKernelPrefix,
                transferredInitializerMap));
        }

        return result;
    }

} // namespace Dml
