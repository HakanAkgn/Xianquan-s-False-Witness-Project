"""Random-function tasks and exactly checked answer-preserving interventions.

A world has a country map g and R person-to-company maps f_r.  Table
entries encode the outputs; their absolute positions encode the inputs.
All random draws use a caller-supplied torch.Generator.
"""
from dataclasses import dataclass, replace
import torch


@dataclass(frozen=True)
class Task:
    n: int = 8
    relations: int = 1
    layout: str = "legacy"
    alpha: float = 0.75
    hint_probability: float = 0.5

    def __post_init__(self):
        if self.n < 4 or self.relations not in (1, 2):
            raise ValueError("Require n>=4 and one or two relations")
        if self.layout not in ("legacy", "fixed"):
            raise ValueError("layout must be legacy or fixed")
        if not (0 <= self.alpha <= 1 and 0 <= self.hint_probability <= 1):
            raise ValueError("probabilities must be in [0,1]")

    @property
    def length(self):
        return (1 + self.relations) * self.n + 3 + (self.relations == 2)

    @property
    def vocab(self):
        return self.n + 4 + (2 if self.relations == 2 else 0)


@dataclass
class World:
    g: torch.Tensor
    f: torch.Tensor  # [batch, relation, person]
    x: torch.Tensor
    relation: torch.Tensor
    two_step: torch.Tensor
    hint: torch.Tensor
    present: torch.Tensor

    def copy(self):
        return World(**{k: v.clone() for k, v in vars(self).items()})

    @property
    def batch(self):
        return self.x.shape[0]

    def intermediate(self):
        b = torch.arange(self.batch)
        return self.f[b, self.relation, self.x]

    def labels(self):
        b = torch.arange(self.batch)
        middle = self.intermediate()
        return torch.where(self.two_step, self.g[b, middle], middle)

    def all_country_answers(self):
        return torch.gather(self.g[:, None, :].expand_as(self.f), 2, self.f)


def draw(task: Task, batch: int, rng: torch.Generator) -> World:
    n, r = task.n, task.relations
    f = torch.randint(n, (batch, r, n), generator=rng)
    g = torch.randint(n, (batch, n), generator=rng)
    x = torch.randint(n, (batch,), generator=rng)
    relation = torch.randint(r, (batch,), generator=rng)
    two = torch.rand(batch, generator=rng) < 0.5
    middle = f[torch.arange(batch), relation, x]
    wrong = (middle + torch.randint(1, n, (batch,), generator=rng)) % n
    hint = torch.where(torch.rand(batch, generator=rng) < task.alpha, middle, wrong)
    present = torch.rand(batch, generator=rng) < task.hint_probability
    return World(g, f, x, relation, two, hint, present)


def encode(world: World, task: Task) -> torch.Tensor:
    h = torch.where(world.present, world.hint, task.n)
    t = task.n + 2 + world.two_step.long()
    suffix = torch.stack((world.x, t, h), 1)
    if task.layout == "legacy":
        suffix[~world.present, 0] = task.n
        suffix[~world.present, 2] = world.x[~world.present]
    pieces = [world.g, world.f.flatten(1)]
    if task.relations == 2:
        pieces.append((task.n + 4 + world.relation)[:, None])
    pieces.append(suffix)
    return torch.cat(pieces, 1)


def equal_function_pair(task: Task, batch: int, rng: torch.Generator,
                        witnesses: int = 1) -> tuple[World, World, torch.Tensor]:
    """Original-style one-relation edit or typed-relation exchange.

    Uses rejection-free conditional construction: three distinct companies a,h,r;
    g(h)=g(r) != g(a); the queried person's selected employer is a.  All other
    entries in the selected relation initially avoid h.  Chosen irrelevant
    rows have r in the queried relation and, in the typed task, h in the other
    relation.  Exchange r/h at those rows.  Row indices and query relation are
    randomized. The label functions g o f_r are preserved for EVERY person and
    EVERY relation, not just for the query. Typed exchange also preserves the
    complete input-token multiset. This conditioning differs from i.i.d. train.
    """
    if not 0 <= witnesses <= task.n - 1:
        raise ValueError("witnesses must be between 0 and n-1")
    w = draw(task, batch, rng)
    n = task.n
    b = torch.arange(batch)
    companies = torch.rand(batch, n, generator=rng).argsort(1)[:, :3]
    a, h, r = companies.unbind(1)
    yw = torch.randint(n, (batch,), generator=rng)
    yt = (yw + torch.randint(1, n, (batch,), generator=rng)) % n
    w.g[b, h] = yw
    w.g[b, r] = yw
    w.g[b, a] = yt
    # Exclude h so the introduced count is known exactly, without rejection.
    selected = (h[:, None] + torch.randint(1, n, (batch, n), generator=rng)) % n
    w.f[b, w.relation] = selected
    w.f[b, w.relation, w.x] = a
    row_scores = torch.rand(batch, n, generator=rng)
    row_scores[b, w.x] = 2.0
    rows = row_scores.argsort(1)[:, :witnesses]
    for k in range(witnesses):
        j = rows[:, k]
        w.f[b, w.relation, j] = r
        if task.relations == 2:
            w.f[b, 1 - w.relation, j] = h
    w.hint = h
    w.present.fill_(True)
    w.two_step.fill_(True)
    v = w.copy()
    for k in range(witnesses):
        j = rows[:, k]
        v.f[b, w.relation, j] = h
        if task.relations == 2:
            v.f[b, 1 - w.relation, j] = r
    validate_pair(w, v, task, rows)
    return w, v, rows


def validate_pair(a: World, b: World, task: Task, rows: torch.Tensor):
    if not torch.equal(a.all_country_answers(), b.all_country_answers()):
        raise AssertionError("An answer function changed")
    if not torch.equal(a.labels(), b.labels()):
        raise AssertionError("The query answer changed")
    if rows.numel() and torch.any(rows == a.x[:, None]):
        raise AssertionError("Intervention touched the queried person")
    if not torch.equal(a.intermediate(), b.intermediate()):
        raise AssertionError("Queried intermediate changed")
    if task.relations == 2:
        if not torch.equal(encode(a, task).sort(1).values,
                           encode(b, task).sort(1).values):
            raise AssertionError("Token multiset changed")


def relevant_change(w: World, task: Task) -> World:
    """Change the queried relation to h; its answer must change on pair worlds.

    Changing the OTHER relation at the same person is an additional negative
    control, implemented by the evaluator. This operation is not an invariance.
    """
    v = w.copy()
    v.f[torch.arange(w.batch), w.relation, w.x] = w.hint
    if torch.any(v.labels() == w.labels()):
        raise ValueError("relevant_change requires decisive pair worlds")
    return v


def balanced_original_pair(task: Task, batch: int, rng: torch.Generator,
                           witnesses: int = 1):
    """Token-balanced intervention accepted by the original 19-token model.

    Change f(j):r->h and compensate g(d):h->r at a company d outside
    image(f). Require g(r)=g(h), j!=x. All composed person answers and
    literal token counts are identical. Neither compensating country symbol
    is the correct or suggested answer. g itself DOES change on unused
    companies; this is not an invariance for company-country questions.
    """
    if task.relations!=1 or not 1<=witnesses<=task.n-3:
        raise ValueError('Require one relation and 1<=witnesses<=n-3')
    n=task.n
    w=draw(task,batch,rng)
    idx=torch.arange(batch)
    order=torch.rand(batch,n,generator=rng).argsort(1)
    a,h,r=order[:,:3].unbind(1)
    dummy=order[:,3:3+witnesses]
    # Uniformly choose two different answer symbols, neither equal to h or r.
    scores=torch.rand(batch,n,generator=rng)
    scores[idx,h]=2.;scores[idx,r]=2.
    answers=scores.argsort(1)[:,:2]
    yw,yt=answers.unbind(1)
    w.g[idx,h]=yw;w.g[idx,r]=yw;w.g[idx,a]=yt
    allowed=torch.ones(batch,n,dtype=torch.bool)
    allowed[idx,h]=False
    allowed.scatter_(1,dummy,False)
    pool=torch.arange(n).expand(batch,n)[allowed].reshape(batch,n-witnesses-1)
    choices=torch.randint(pool.shape[1],(batch,n),generator=rng)
    w.f[:,0]=torch.gather(pool,1,choices)
    w.f[idx,0,w.x]=a
    row_scores=torch.rand(batch,n,generator=rng);row_scores[idx,w.x]=2.
    rows=row_scores.argsort(1)[:,:witnesses]
    for k in range(witnesses):
        w.f[idx,0,rows[:,k]]=r
        w.g[idx,dummy[:,k]]=h
    w.hint=h;w.present.fill_(True);w.two_step.fill_(True)
    v=w.copy()
    for k in range(witnesses):
        v.f[idx,0,rows[:,k]]=h
        v.g[idx,dummy[:,k]]=r
    validate_pair(w,v,task,rows)
    if not torch.equal(encode(w,task).sort(1).values,encode(v,task).sort(1).values):
        raise AssertionError('Balanced original intervention changed token counts')
    for k in range(witnesses):
        if torch.any(v.f[:,0]==dummy[:,k,None]):
            raise AssertionError('Compensation company is used')
    return w,v,rows,dummy
