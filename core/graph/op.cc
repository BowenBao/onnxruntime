#include "op.h"
#include "utils.h"

namespace LotusIR
{
    InferenceContext::InferenceContext(const Node* p_node,
        const OperatorSchema* p_opSchema,
        const std::vector<NodeArg>* p_inputs,
        std::vector<NodeArg>* p_outputs)
        : m_node(p_node),
        m_opSchema(p_opSchema),
        m_inputs(p_inputs),
        m_outputs(p_outputs)
    {
    }

    const Node* InferenceContext::GetNode() const
    {
        return m_node;
    }

    const OperatorSchema* InferenceContext::GetOp() const
    {
        return m_opSchema;
    }

    const std::vector<NodeArg>* InferenceContext::GetInputs() const
    {
        return m_inputs;
    }

    std::vector<NodeArg>* InferenceContext::Mutable_Outputs()
    {
        return m_outputs;
    }

    OperatorSchemaSetter&
        OperatorSchemaSetter::Name(const std::string& p_opName)
    {
        m_name = p_opName;
        return *this;
    }

    OperatorSchemaSetter&
        OperatorSchemaSetter::Description(const std::string& p_description)
    {
        m_description = p_description;
        return *this;
    }

    OperatorSchemaSetter& 
        OperatorSchemaSetter::Input(const std::string& p_inputName,
                                    const std::string& p_type,
                                    const std::string& p_description)
    {
        m_inputs.push_back(std::make_tuple(p_inputName, p_type, p_description));
        return *this;
    }

    OperatorSchemaSetter& 
        OperatorSchemaSetter::Output(const std::string& p_outputName,
                                     const std::string& p_type,
                                     const std::string& p_description)
    {
        m_outputs.push_back(std::make_tuple(p_outputName, p_type, p_description));
        return *this;
    }

    OperatorSchemaSetter&
        OperatorSchemaSetter::Attr(const std::string& p_attrName,
                                   AttrType p_attrType,
                                   const std::string& p_description)
    {
        m_attributes.push_back(make_tuple(p_attrName, p_attrType, p_description, AttributeProto()));
        return *this;
    }

    OperatorSchemaSetter&
        OperatorSchemaSetter::Attr(const std::string& p_attrName,
                                   AttrType p_attrType,
                                   const std::string& p_description,
                                   const int64_t& p_defaultValue)
    {
        AttributeProto a;
        a.set_name(p_attrName);
        a.set_i(p_defaultValue);
        m_attributes.push_back(make_tuple(p_attrName, p_attrType, p_description, a));
        return *this;
    }

    OperatorSchemaSetter&
        OperatorSchemaSetter::Attr(const std::string& p_attrName,
                                   AttrType p_attrType,
                                   const std::string& p_description,
                                   const std::vector<int64_t>& p_defaultValue)
    {
        AttributeProto a;
        a.set_name(p_attrName);
        for (const auto& v : p_defaultValue)
        {
            a.add_ints(v);
        }
        m_attributes.push_back(make_tuple(p_attrName, p_attrType, p_description, a));
        return *this;
    }

    OperatorSchemaSetter&
        OperatorSchemaSetter::Attr(const std::string& p_attrName,
                                   AttrType p_attrType,
                                   const std::string& p_description,
                                   const float& p_defaultValue)
    {
        AttributeProto a;
        a.set_name(p_attrName);
        a.set_f(p_defaultValue);
        m_attributes.push_back(make_tuple(p_attrName, p_attrType, p_description, a));
        return *this;
    }

    OperatorSchemaSetter&
        OperatorSchemaSetter::Attr(const std::string& p_attrName,
                                   AttrType p_attrType,
                                   const std::string& p_description,
                                   const std::vector<float>& p_defaultValue)
    {
        AttributeProto a;
        a.set_name(p_attrName);
        for (const auto& v : p_defaultValue)
        {
            a.add_floats(v);
        }
        m_attributes.push_back(make_tuple(p_attrName, p_attrType, p_description, a));
        return *this;
    }

    OperatorSchemaSetter&
        OperatorSchemaSetter::Attr(const std::string& p_attrName,
                                   AttrType p_attrType,
                                   const std::string& p_description,
                                   const std::string& p_defaultValue)
    {
        AttributeProto a;
        a.set_name(p_attrName);
        a.set_s(p_defaultValue);
        m_attributes.push_back(make_tuple(p_attrName, p_attrType, p_description, a));
        return *this;
    }

    OperatorSchemaSetter&
        OperatorSchemaSetter::Attr(const std::string& p_attrName,
                                   AttrType p_attrType,
                                   const std::string& p_description,
                                   const std::vector<std::string>& p_defaultValue)
    {
        AttributeProto a;
        a.set_name(p_attrName);
        for (const auto& v : p_defaultValue)
        {
            a.add_strings(v);
        }
        m_attributes.push_back(make_tuple(p_attrName, p_attrType, p_description, a));
        return *this;
    }

    OperatorSchemaSetter&
        OperatorSchemaSetter::TypeConstraint(const std::string& p_typeName,
                                             const std::vector<std::string>& p_constraints,
                                             const std::string& p_description)
    {
        m_constraints.push_back(std::make_tuple(p_typeName, p_constraints, p_description));
        return *this;
    }

    OperatorSchemaSetter&
        OperatorSchemaSetter::SetShapeInferenceFunc(
            ShapeInferenceFunc p_shapeInferFunc)
    {
        m_shapeInferFunc = p_shapeInferFunc;
        return *this;
    }

    OperatorSchemaSetter&
        OperatorSchemaSetter::SetAttributeParser(
            AttributeParser p_attrParser)
    {
        m_parser = p_attrParser;
        return *this;
    }

    OperatorSchema::FormalParameter::FormalParameter(
        const std::string& p_name, const std::string& p_type,
        const std::string& p_description,
        const TypeConstraintMap& p_constraintMap)
        : m_name(p_name), m_typeStr(p_type), m_description(p_description)
    {
        auto it = p_constraintMap.find(p_type);
        if (it != p_constraintMap.end())
        {
            m_types = it->second.first;
        }
        else
        {
            m_types.emplace(Utils::OpUtils::ToType(m_typeStr));
        }
    }

    const std::string& OperatorSchema::FormalParameter::GetName() const
    {
        return m_name;
    }

    const DataTypeSet& OperatorSchema::FormalParameter::GetTypes() const
    {
        return m_types;
    }

    const std::string& OperatorSchema::FormalParameter::GetTypeStr() const
    {
        return m_typeStr;
    }

    const std::string& OperatorSchema::FormalParameter::GetDescription() const
    {
        return m_description;
    }

    OperatorSchema::Attribute::Attribute(
        const std::string& p_attrName,
        AttrType p_type,
        const std::string& p_description,
        const AttributeProto& p_defaultVal)
        : m_name(p_attrName), m_type(p_type), m_description(p_description),
          m_hasDefaultValue(true)
    {
        m_allowedValues.push_back(p_defaultVal);
    }

    OperatorSchema::Attribute::Attribute(
        const std::string& p_attrName,
        AttrType p_type,
        const std::string& p_description)
        : m_name(p_attrName), m_type(p_type), m_description(p_description),
        m_hasDefaultValue(false)
    {
    }

    const std::string& OperatorSchema::Attribute::GetName() const
    {
        return m_name;
    }

    AttrType OperatorSchema::Attribute::GetType() const
    {
        return m_type;
    }

    bool OperatorSchema::Attribute::HasDefaultValue(
        const AttributeProto** p_value) const
    {
        if (m_hasDefaultValue
            && nullptr != p_value)
        {
            *p_value = &(m_allowedValues[0]);
        }

        return m_hasDefaultValue;
    }

    OperatorSchema::OperatorSchema(const OperatorSchemaSetter& p_setter)
        : m_name(p_setter.m_name),
        m_description(p_setter.m_description),
        m_shapeInferFunc(p_setter.m_shapeInferFunc),
        m_parser(p_setter.m_parser)
    {
        // Process type constraints.
        for (const auto& constraint : p_setter.m_constraints)
        {
            std::string name;
            std::vector<std::string> types;
            std::string desc;
            std::tie(name, types, desc) = constraint;

            auto it = m_typeConstraintMap.find(name);
            if (it == m_typeConstraintMap.end())
            {
                DataTypeSet d;
                for (const auto& t : types)
                {
                    d.insert(Utils::OpUtils::ToType(t));
                }
                m_typeConstraintMap.insert(std::make_pair(name, std::make_pair(d, desc)));
            }
            else
            {
                // already a constraint with the same name. error.
            }
        }

        m_inputs.reserve(p_setter.m_inputs.size());
        for (const auto& input : p_setter.m_inputs)
        {
            std::string name;
            std::string type;
            std::string desc;
            std::tie(name, type, desc) = input;
            m_inputs.push_back(FormalParameter(name, type, desc, m_typeConstraintMap));
        }

        m_outputs.reserve(p_setter.m_outputs.size());
        for (const auto& output : p_setter.m_outputs)
        {
            std::string name;
            std::string type;
            std::string desc;
            std::tie(name, type, desc) = output;
            m_outputs.push_back(FormalParameter(name, type, desc, m_typeConstraintMap));
        }

        m_attributes.reserve(p_setter.m_attributes.size());
        for (const auto& attr : p_setter.m_attributes)
        {
            std::string name;
            AttrType type;
            std::string desc;
            AttributeProto a;
            std::tie(name, type, desc, a) = attr;
            if (a.name() == name)
            {
                m_attributes.push_back(Attribute(name, type, desc, a));
            }
            else
            {
                m_attributes.push_back(Attribute(name, type, desc));
            }
        }
    }

    const std::string& OperatorSchema::GetName() const
    {
        return m_name;
    }

    const std::string& OperatorSchema::GetDescription() const
    {
        return m_description;
    }

    const std::vector<OperatorSchema::FormalParameter>&
        OperatorSchema::GetInputs() const
    {
        return m_inputs;
    }

    const std::vector<OperatorSchema::FormalParameter>&
        OperatorSchema::GetOutputs() const
    {
        return m_outputs;
    }

    const std::vector<OperatorSchema::Attribute>&
        OperatorSchema::GetAttributes() const
    {
        return m_attributes;
    }

    ShapeInferenceFunc OperatorSchema::GetShapeInferenceFunc() const
    {
        return m_shapeInferFunc;
    }

    AttributeParser OperatorSchema::GetAttributeParser() const
    {
        return m_parser;
    }

    const TypeConstraintMap& OperatorSchema::GetTypeConstraintMap() const
    {
        return m_typeConstraintMap;
    }

    OperatorSchemaRegistry::RegisterOnce::RegisterOnce(
        const OperatorSchemaSetter& p_opSchema)
    {
        OperatorSchemaRegistry::Get()->Register(p_opSchema);
    }

    bool OperatorSchemaRegistry::TryGetOp(const std::string& p_name,
        const OperatorSchema** p_opRegistry) const
    {
        if (nullptr == p_opRegistry)
        {
            return false;
        }

        auto iter = m_operatorRegistryMap.find(p_name);
        if (m_operatorRegistryMap.end() == iter)
        {
            return false;
        }
        *p_opRegistry = &(iter->second);
        return true;
    }

    Status OperatorSchemaRegistry::Register(
        const OperatorSchema& p_opSchema)
    {
        auto iter = m_operatorRegistryMap.find(p_opSchema.GetName());
        if (m_operatorRegistryMap.end() != iter)
        {
            Status status(false,
                "Error: operator schema with same name ("
                + p_opSchema.GetName() + ") exists.");
            return status;
        }
        else
        {
            m_operatorRegistryMap[p_opSchema.GetName()] = p_opSchema;
            return Status::OK();
        }
    }

    OperatorSchemaRegistry* OperatorSchemaRegistry::Get()
    {
        static OperatorSchemaRegistry* s_registry
            = new OperatorSchemaRegistry();
        return s_registry;
    }
}
