using System;

namespace FunctorialMembrane
{
    public abstract class PrimitiveCarrier
    {
        private PrimitiveCarrier() { }

        public sealed class T_P : PrimitiveCarrier
        {
            public object Payload { get; }
            public T_P(object payload) { Payload = payload; }
        }

        public sealed class T_C : PrimitiveCarrier
        {
            public object Payload { get; }
            public T_C(object payload) { Payload = payload; }
        }

        public sealed class T_H : PrimitiveCarrier
        {
            public object Payload { get; }
            public T_H(object payload) { Payload = payload; }
        }
    }

    public static class Membrane
    {
        public static PrimitiveCarrier.T_P Passage(
            PrimitiveCarrier.T_P carrier,
            Func<object, object> transform)
        {
            return new PrimitiveCarrier.T_P(transform(carrier.Payload));
        }

        public static PrimitiveCarrier.T_C Passage(
            PrimitiveCarrier.T_C carrier,
            Func<object, object> transform)
        {
            return new PrimitiveCarrier.T_C(transform(carrier.Payload));
        }

        public static PrimitiveCarrier.T_H Passage(
            PrimitiveCarrier.T_H carrier,
            Func<object, object> transform)
        {
            return new PrimitiveCarrier.T_H(transform(carrier.Payload));
        }
    }
}
