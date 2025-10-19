using Galileu.Node.Core;
using Galileu.Node.Interfaces;

namespace Galileu.Node.Brain;

public class LstmStepCache
{
    public IMathTensor? Input { get; set; }
    public IMathTensor? HiddenPrev { get; set; }
    public IMathTensor? CellPrev { get; set; }
    public IMathTensor? ForgetGate { get; set; }
    public IMathTensor? InputGate { get; set; }
    public IMathTensor? CellCandidate { get; set; }
    public IMathTensor? OutputGate { get; set; }
    public IMathTensor? CellNext { get; set; }
    public IMathTensor? TanhCellNext { get; set; }
    public IMathTensor? HiddenNext { get; set; }
}