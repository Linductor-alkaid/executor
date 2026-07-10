#pragma once

namespace executor::comm {

template <class T>
class MpscChannel;

template <class T>
using SpscChannel = MpscChannel<T>;

template <class T>
class LatestMailbox;

template <class T>
class RealtimeChannel;

class PhaseGate;
class Sequencer;

template <class T>
struct Snapshot;

template <class T>
class DoubleBuffer;

} // namespace executor::comm
