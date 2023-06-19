// Copyright (c) 2023 Graphcore Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <boost/range/algorithm.hpp>

#include <popart/ir.hpp>
#include <popart/op.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/op/loop.hpp>
#include <popart/scheduler_requireoptimal.hpp>
#include <popart/tensor.hpp>
#include <popart/transforms/transform.hpp>

#include "new_ipu_copy.hpp"

namespace popart
{
  class Graph;

  /*
              src_tensor                               src_tensor
            /            \                                 |
          ipuCopy      ipuCopy     ---------->         newIpuCopy
            |              |                             /     \
          dst_1          dst_2                        dst_1    dst_2
  */
  class MergeCopiseWithSameSrc : public Transform
  {
  public:
    static std::size_t id();

    MergeCopiseWithSameSrc() : Transform() {}
    virtual ~MergeCopiseWithSameSrc() override {}

    virtual bool apply(Graph &graph) const final;

    virtual std::size_t getId() const final { return id(); }

    virtual std::string getName() const final { return "MergeCopiseWithSameSrc"; }
  };

  std::size_t MergeCopiseWithSameSrc::id() { return typeid(MergeCopiseWithSameSrc).hash_code(); }

  static NewIpuCopyOp *createNewCopyOp(Graph &graph, uint64_t from_ipu)
  {
    DebugInfo di({"NewIpuCopyOp"}, "popartbuilder");

    auto op_id = popart::OperatorIdentifier{"custom.ops", "NewIpuCopy", 1};
    Op::Settings settings(graph, "", di.getId());
    auto newIpuCopy_op =
        std::make_unique<NewIpuCopyOp>(op_id, from_ipu, settings);
    auto newIpuCopy = newIpuCopy_op.get();
    graph.moveIntoGraph(std::move(newIpuCopy_op));
    return newIpuCopy;
  }

  static uint64_t getSrcIpu(const std::vector<Tensor *> &copy_group)
  {
    // check src ipu same
    auto p = copy_group.front()->getProducer();
    return dynamic_cast<IpuCopyOp *>(p)->getSourceIpu();
  }

  static void mergeCopies(const std::vector<Tensor *> &copy_group, Graph &graph)
  {
    auto getSourceTensor = [](Tensor *t)
    {
      // assumes producer of t has 1 input only.
      return t->getProducer()->input->tensor(0);
    };

    // create a new copy op
    auto src_ipu = getSrcIpu(copy_group);

    // Create New Copy Op in the subgraph of copy group

    auto new_copy_op = createNewCopyOp(graph, src_ipu);

    new_copy_op->setVirtualGraphId(src_ipu);

    // Get the execution context of the copy group (assuming all have the same)
    new_copy_op->settings.executionContext =
        copy_group.back()->getProducer()->settings.executionContext;

    // move the copies
    for (auto t : copy_group)
    {
      auto source = getSourceTensor(t);
      auto producer = dynamic_cast<IpuCopyOp *>(t->getProducer());
      auto dest_ipu = producer->getDestIpu();

      if (producer->input->n() != 1)
      {
        throw internal_error(
            "Attempting to merge a copy with more than one input!");
      }

      producer->disconnectInTensor(0, source);
      producer->disconnectOutTensor(t);

      int idx = new_copy_op->output->n();
      new_copy_op->connectInTensor(idx, source->id, src_ipu);
      new_copy_op->connectOutTensor(idx, t->id, dest_ipu);

      graph.eraseOp(producer->id);
    }
    new_copy_op->setup();
  }

  static bool isMultipleCopiesCopySrc(const Tensor *t)
  {
    const auto consumers = t->consumers;
    if (consumers.getTotal() > 1)
    {
      const auto consumers = t->consumers;
      int num_copies = 0;
      for (auto op : consumers.getOps())
      {
        if (op->isConvertibleTo<IpuCopyOp>())
        {
          num_copies++;
        }
        if (num_copies > 1)
        {
          return true;
        }
      }
    }
    return false;
  }

  static std::vector<Op *> getOpsThatProduceMultipleCopies(Graph &graph)
  {
    std::vector<Op *> ops;

    for (auto &id_op : graph.getOps())
    {
      auto op = id_op.second.get();
      if (boost::count_if(op->output->tensors(), isMultipleCopiesCopySrc) >= 1)
      {
        ops.push_back(op);
      }
    }

    return ops;
  }

  // check that the op at position `op_schedule_iter`
  // is the first consumer of `tensor` to appear in `op_schedule`
  template <typename T>
  static bool checkOpIsFirstConsumer(const T &op_schedule_iter, Tensor *tensor,
                                     const std::vector<Op *> &op_schedule)
  {
    for (auto consumer : tensor->consumers.getOps())
    {
      if (std::find(op_schedule.begin(), op_schedule_iter, consumer) !=
          op_schedule_iter)
      {
        return false;
      }
    }

    return true;
  }

  static std::vector<Tensor *>
  createCopyGroup(Op *op, const std::vector<Op *> &op_schedule)
  {
    std::vector<Tensor *> group;
    const auto op_schedule_iter =
        std::find(op_schedule.begin(), op_schedule.end(), op);

    for (auto tensor : op->output->tensors())
    {
      if (isMultipleCopiesCopySrc(tensor))
      {
        for (auto consumer : tensor->consumers.getOps())
        {
          if (consumer->input->n() == 1 && consumer->output->n() == 1 &&
              checkOpIsFirstConsumer(op_schedule_iter, tensor, op_schedule))
          {
            group.push_back(consumer->output->tensors()[0]);
          }
        }
      }
    }
    return group;
  }

  bool MergeCopiseWithSameSrc::apply(Graph &graph) const
  {
    for (auto &id_op : graph.getOps())
    {
      auto op = id_op.second.get();
      if (LoopOp *loopOp = dynamic_cast<LoopOp *>(op))
      {
        Graph &calledGraph = loopOp->getCalledGraph();
        apply(calledGraph);
      }
    }

    const auto multiple_copy_producers = getOpsThatProduceMultipleCopies(graph);
    const auto op_schedule = graph.getOpSchedule({}, RequireOptimalSchedule::Yes);
    for (auto op : multiple_copy_producers)
    {
      const auto copy_group = createCopyGroup(op, op_schedule);
      if (copy_group.size() > 1)
      {

        // Skip if not all copies in the group have the same execution context
        bool allSameExecutionContext = true;
        auto executionConstext0 =
            copy_group.back()->getProducer()->settings.executionContext;
        for (auto t : copy_group)
        {
          if (t->getProducer()->settings.executionContext != executionConstext0)
          {
            allSameExecutionContext = false;
          }
        }
        if (!allSameExecutionContext)
        {
          continue;
        }

        // without pipelining, we merge copies with different destiantions
        if (!graph.getIr().getSessionOptions().enablePipelining)
        {
          mergeCopies(copy_group, graph);
        }
      }
    }
    return true;
  }

  namespace
  {
    bool init = Transform::registerTransform(new MergeCopiseWithSameSrc);
  }

} // namespace popart
