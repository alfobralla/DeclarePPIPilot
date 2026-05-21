import { useEffect, useMemo, useRef, useState } from 'react'
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

const PROVIDERS = [
  { value: 'openai', label: 'OpenAI' },
  { value: 'anthropic', label: 'Anthropic' },
  { value: 'lmstudio', label: 'Local (LM Studio)' },
]

function parseDurationLike(raw) {
  if (raw == null) return Number.NaN
  const s = String(raw).trim()
  if (!s) return Number.NaN

  const direct = Number(s)
  if (!Number.isNaN(direct)) return direct

  const m = s.match(/^(?:(-?\d+)\s+days?,?\s+)?(\d{1,2}):(\d{2}):(\d{2})(?:\.(\d+))?$/i)
  if (m) {
    const days = Number(m[1] || 0)
    const hh = Number(m[2] || 0)
    const mm = Number(m[3] || 0)
    const ss = Number(m[4] || 0)
    const frac = m[5] ? Number(`0.${m[5]}`) : 0
    return days * 86400 + hh * 3600 + mm * 60 + ss + frac
  }

  const td = s.match(/^Timedelta\('(.*)'\)$/)
  if (td) return parseDurationLike(td[1])

  const iso = s.match(/^(-)?P(?:(\d+)D)?T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?$/i)
  if (iso) {
    const sign = iso[1] ? -1 : 1
    const days = Number(iso[2] || 0)
    const hours = Number(iso[3] || 0)
    const minutes = Number(iso[4] || 0)
    const seconds = Number(iso[5] || 0)
    return sign * (days * 86400 + hours * 3600 + minutes * 60 + seconds)
  }

  return Number.NaN
}

function isDurationLike(raw) {
  if (raw == null) return false
  if (typeof raw !== 'string') return false
  const s = raw.trim()
  if (!s) return false
  if (/^Timedelta\('(.*)'\)$/.test(s)) return true
  if (/^(?:(-?\d+)\s+days?,?\s+)?(\d{1,2}):(\d{2}):(\d{2})(?:\.(\d+))?$/i.test(s)) return true
  if (/^(-)?P(?:(\d+)D)?T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?$/i.test(s)) return true
  return false
}

function toNumericValue(raw) {
  if (typeof raw === 'number') return Number.isFinite(raw) ? raw : Number.NaN
  return parseDurationLike(raw)
}

function toCvNumber(value) {
  const n = Number(value)
  return Number.isFinite(n) ? n : null
}

function sanitizeId(value) {
  return String(value).replace(/[^a-zA-Z0-9_-]/g, '_')
}

function goalSectionId(goalName) {
  return `goal-${sanitizeId(goalName).toLowerCase()}`
}

function formatPeriodLabel(freqRaw) {
  if (!freqRaw) return 'none'
  const m = String(freqRaw).trim().match(/^(\d+)(WE|ME|YE)$/i)
  if (!m) return String(freqRaw)
  const qty = Number(m[1])
  const unit = m[2].toUpperCase()
  if (unit === 'WE') return `${qty} ${qty === 1 ? 'week' : 'weeks'}`
  if (unit === 'ME') return `${qty} ${qty === 1 ? 'month' : 'months'}`
  if (unit === 'YE') return `${qty} ${qty === 1 ? 'year' : 'years'}`
  return String(freqRaw)
}

function parseGroupedValues(groupedValues) {
  let source = groupedValues
  if (typeof source === 'string') {
    const txt = source.trim()
    if (!txt) return null
    try {
      source = JSON.parse(txt)
    } catch {
      try {
        source = JSON.parse(txt.replaceAll("'", '"'))
      } catch {
        return null
      }
    }
  }
  if (!Array.isArray(source) || source.length === 0) return null

  const parsed = source
    .map((row, idx) => {
      if (row && typeof row === 'object' && !Array.isArray(row)) {
        const bucket = row.bucket ?? row.group ?? row.key ?? row.period ?? row.date ?? row.label ?? idx
        const rawValue = row.value ?? row.val ?? row.metric ?? row.computation_value
        const numFromBackend = row.value_numeric
        return {
          bucket: String(bucket),
          rawValue,
          value: Number.isFinite(Number(numFromBackend)) ? Number(numFromBackend) : toNumericValue(rawValue),
        }
      }

      if (Array.isArray(row) && row.length >= 2) {
        return {
          bucket: String(row[0]),
          rawValue: row[1],
          value: toNumericValue(row[1]),
        }
      }

      return {
        bucket: String(idx),
        rawValue: row,
        value: toNumericValue(row),
      }
    })
    .filter((x) => x.bucket.length > 0)

  if (parsed.length === 0 || parsed.some((x) => Number.isNaN(x.value))) return null
  return parsed
}

function formatDurationFromSeconds(totalSecondsRaw) {
  const numeric = Number(totalSecondsRaw)
  if (!Number.isFinite(numeric)) return String(totalSecondsRaw)
  const sign = numeric < 0 ? '-' : ''
  let total = Math.abs(numeric)
  const days = Math.floor(total / 86400)
  total -= days * 86400
  const hours = Math.floor(total / 3600)
  total -= hours * 3600
  const minutes = Math.floor(total / 60)
  total -= minutes * 60
  const seconds = total
  if (days > 0) return `${sign}${days}d ${hours}h ${minutes}m`
  if (hours > 0) return `${sign}${hours}h ${minutes}m`
  if (minutes > 0) return `${sign}${minutes}m ${seconds.toFixed(0)}s`
  return `${sign}${seconds.toFixed(2)}s`
}

function formatMetricValue(raw, options = {}) {
  const { forceDuration = false } = options
  if (raw == null) return 'Not available'

  const shouldFormatAsDuration = forceDuration || isDurationLike(raw)
  const numeric = toNumericValue(raw)
  if (!Number.isNaN(numeric)) {
    if (shouldFormatAsDuration) {
      return formatDurationFromSeconds(numeric)
    }
    const abs = Math.abs(numeric)
    if (abs >= 1000000) return numeric.toLocaleString(undefined, { maximumFractionDigits: 1 })
    if (abs >= 1000) return numeric.toLocaleString(undefined, { maximumFractionDigits: 2 })
    if (abs >= 1) return numeric.toLocaleString(undefined, { maximumFractionDigits: 3 })
    return numeric.toLocaleString(undefined, { maximumFractionDigits: 6 })
  }

  return String(raw)
}

function statusBadgeClass(status) {
  if (status === 'failed') return 'badge border-rose-300 bg-rose-50 text-rose-700'
  return ''
}

function cvBadgeClass(cv) {
  if (cv == null) return 'badge border-slate-300 bg-slate-100 text-slate-600'
  if (cv < 0.2) return 'badge border-amber-300 bg-amber-50 text-amber-700'
  if (cv < 0.5) return 'badge border-yellow-300 bg-yellow-50 text-yellow-700'
  return 'badge border-emerald-300 bg-emerald-50 text-emerald-700'
}

function SingleValueMetric({ value, isDuration }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-slate-50 px-4 py-3">
      <p className="text-xs uppercase tracking-wide text-slate-500">Computed value</p>
      <p className="mt-2 break-all text-3xl font-extrabold text-slate-900">{formatMetricValue(value, { forceDuration: isDuration })}</p>
    </div>
  )
}

function GroupedValueChart({ groupedValues, timeGrouperFreq, singleValue, isDuration }) {
  const parsed = parseGroupedValues(groupedValues)
  const baseline = toNumericValue(singleValue)
  const hasBaseline = !Number.isNaN(baseline)
  if (!parsed) {
    return (
      <div className="rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-700">
        <p className="font-semibold text-slate-800">Grouped values</p>
        <p className="mt-1">Not available for chart rendering.</p>
      </div>
    )
  }

  return (
    <div className="rounded-xl border border-slate-200 bg-slate-50 px-2 py-3">
      <p className="px-2 text-sm font-semibold text-slate-800">Grouped values ({timeGrouperFreq})</p>
      {hasBaseline ? (
        <p className="px-2 text-xs text-slate-500">
          <span className="inline-block align-middle mr-1 h-0.5 w-6 border-t-2 border-dashed border-slate-400" />
          baseline: {formatMetricValue(singleValue, { forceDuration: isDuration })}
        </p>
      ) : null}
      <div className="h-56 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={parsed} margin={{ top: 12, right: 12, left: 0, bottom: 18 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis
              dataKey="bucket"
              tick={{ fill: '#475569', fontSize: 11 }}
              angle={-30}
              textAnchor="end"
              height={48}
            />
            <YAxis
              tick={{ fill: '#475569', fontSize: 12 }}
              tickFormatter={(v) => (isDuration ? formatDurationFromSeconds(v) : formatMetricValue(v))}
            />
            <Tooltip formatter={(v, _name, ctx) => [formatMetricValue(ctx?.payload?.rawValue ?? v, { forceDuration: isDuration }), 'Value']} />
            {hasBaseline ? (
              <ReferenceLine
                y={baseline}
                stroke="#64748b"
                strokeDasharray="6 4"
                ifOverflow="extendDomain"
              />
            ) : null}
            <Line
              type="monotone"
              dataKey="value"
              stroke="#0f766e"
              strokeWidth={2}
              dot={{ fill: '#0f766e', r: 3 }}
              activeDot={{ r: 5 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <details className="mx-2 mt-2 rounded-lg border border-slate-200 bg-white p-2 text-xs text-slate-700">
        <summary className="cursor-pointer font-semibold text-slate-700">Grouped values table</summary>
        <div className="mt-2 max-h-48 overflow-auto rounded border border-slate-200">
          <table className="min-w-full text-left">
            <thead className="sticky top-0 bg-slate-100">
              <tr>
                <th className="px-2 py-1 font-semibold">Bucket</th>
                <th className="px-2 py-1 font-semibold">Value</th>
              </tr>
            </thead>
            <tbody>
              {parsed.map((row) => (
                <tr key={`${row.bucket}-${row.value}`} className="border-t border-slate-100">
                  <td className="px-2 py-1">{row.bucket}</td>
                  <td className="px-2 py-1">{formatMetricValue(row.rawValue, { forceDuration: isDuration })}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </div>
  )
}

function cloneFormEntries(entries) {
  return entries.map(([key, value]) => [key, value])
}

function buildFormDataFromEntries(entries) {
  const formData = new FormData()
  for (const [k, v] of entries) {
    formData.append(k, v)
  }
  return formData
}

function kpiHasDurationValues(kpi) {
  const singleDuration = isDurationLike(kpi.single_value)
  const grouped = kpi.grouped_values
  if (!Array.isArray(grouped) || grouped.length === 0) return singleDuration
  const groupedDuration = grouped.some((row) => row && typeof row === 'object' && isDurationLike(row.value))
  return singleDuration || groupedDuration
}

function App() {
  const [isRunning, setIsRunning] = useState(false)
  const [error, setError] = useState('')
  const [status, setStatus] = useState('')
  const [data, setData] = useState(null)
  const [showForm, setShowForm] = useState(true)
  const [lastRunEntries, setLastRunEntries] = useState(null)
  const [activeGoal, setActiveGoal] = useState('')
  const [hideLowCv, setHideLowCv] = useState(false)
  const [debugMode, setDebugMode] = useState(false)
  const [sortKpis, setSortKpis] = useState('id_asc')
  const [valueMode, setValueMode] = useState('grouped')
  const [openGoalTooltip, setOpenGoalTooltip] = useState('')
  const contentRef = useRef(null)

  const filteredGoals = useMemo(() => {
    if (!data) return []

    const cvThreshold = data?.meta?.cv_threshold == null ? 0.2 : Number(data.meta.cv_threshold)

    const sorted = (kpis) => {
      const arr = [...kpis]
      if (sortKpis === 'id_asc') return arr.sort((a, b) => String(a.kpi_id).localeCompare(String(b.kpi_id)))
      if (sortKpis === 'id_desc') return arr.sort((a, b) => String(b.kpi_id).localeCompare(String(a.kpi_id)))
      if (sortKpis === 'cv_desc') {
        return arr.sort((a, b) => (toCvNumber(b.coefficient_of_variation) ?? -Infinity) - (toCvNumber(a.coefficient_of_variation) ?? -Infinity))
      }
      if (sortKpis === 'cv_asc') {
        return arr.sort((a, b) => (toCvNumber(a.coefficient_of_variation) ?? Infinity) - (toCvNumber(b.coefficient_of_variation) ?? Infinity))
      }
      if (sortKpis === 'status') {
        const rank = { success: 0, skipped: 1, failed: 2, null: 3 }
        return arr.sort((a, b) => (rank[a.execution_status] ?? 3) - (rank[b.execution_status] ?? 3))
      }
      return arr
    }

    return data.goals
      .map((goal) => {
        const computable = goal.kpis.filter((k) => {
          if (k.execution_status !== 'success') return false
          if (valueMode === 'single') return k.single_value != null
          return k.variability_status === 'ok' && Array.isArray(k.grouped_values) && k.grouped_values.length > 0
        })

        const forDisplay = debugMode ? goal.kpis : computable
        const visible = forDisplay.filter((k) => {
          if (!hideLowCv) return true
          const cv = toCvNumber(k.coefficient_of_variation)
          if (cv == null) return true
          return cv >= cvThreshold
        })
        const shownComputable = visible.filter((k) => computable.some((c) => c.kpi_id === k.kpi_id)).length

        return {
          ...goal,
          shown_kpis: sorted(visible),
          total_kpis: computable.length,
          shown_computable_kpis: shownComputable,
        }
      })
      .filter((goal) => goal.shown_kpis.length > 0)
  }, [data, debugMode, hideLowCv, sortKpis, valueMode])

  useEffect(() => {
    if (filteredGoals.length === 0) {
      setActiveGoal('')
      return
    }

    const ids = filteredGoals.map((g) => goalSectionId(g.goal_name))
    if (!activeGoal || !ids.includes(activeGoal)) {
      setActiveGoal(ids[0])
    }

    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((entry) => entry.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio)
        if (visible.length > 0) {
          setActiveGoal(visible[0].target.id)
        }
      },
      { rootMargin: '-96px 0px -55% 0px', threshold: [0.1, 0.25, 0.5] }
    )

    ids.forEach((id) => {
      const el = document.getElementById(id)
      if (el) observer.observe(el)
    })

    return () => observer.disconnect()
  }, [filteredGoals, activeGoal])

  const summary = useMemo(() => {
    if (!data) return null
    const shown = filteredGoals.reduce((acc, g) => acc + g.shown_kpis.length, 0)
    return {
      totalKPIs: data.meta.n_kpis,
      shownKPIs: shown,
      totalGoals: data.meta.n_goals,
      shownGoals: filteredGoals.length,
      period: data.meta.time_grouper_freq || 'none',
      cvThreshold: data.meta.cv_threshold,
    }
  }, [data, filteredGoals])

  const executeRun = async (formData, entriesSnapshot = null) => {
    setIsRunning(true)
    setError('')
    setStatus('Running full pipeline. This can take a few minutes...')

    try {
      const res = await fetch('/api/run', {
        method: 'POST',
        body: formData,
      })
      const payload = await res.json()
      if (!res.ok) {
        throw new Error(payload.detail || 'Request failed')
      }

      setData(payload)
      setShowForm(false)
      if (entriesSnapshot) {
        setLastRunEntries(cloneFormEntries(entriesSnapshot))
      }
      setStatus(`Run completed: ${payload.meta.n_kpis} KPIs in ${payload.meta.n_goals} goal categories.`)
    } catch (e) {
      setError(e.message || String(e))
      setStatus('')
    } finally {
      setIsRunning(false)
    }
  }

  const onSubmit = async (event) => {
    event.preventDefault()
    const formData = new FormData(event.currentTarget)
    const entries = [...formData.entries()]
    await executeRun(buildFormDataFromEntries(entries), entries)
  }

  const onRunAgain = async () => {
    if (!lastRunEntries || isRunning) return
    const entries = cloneFormEntries(lastRunEntries)
    await executeRun(buildFormDataFromEntries(entries), entries)
  }

  const scrollToGoal = (goalName) => {
    const id = goalSectionId(goalName)
    const el = document.getElementById(id)
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'start' })
      setActiveGoal(id)
    }
  }

  useEffect(() => {
    const onDocClick = () => setOpenGoalTooltip('')
    const onEsc = (e) => {
      if (e.key === 'Escape') setOpenGoalTooltip('')
    }
    document.addEventListener('click', onDocClick)
    document.addEventListener('keydown', onEsc)
    return () => {
      document.removeEventListener('click', onDocClick)
      document.removeEventListener('keydown', onEsc)
    }
  }, [])

  return (
    <div className="min-h-screen w-full">
      <header className="sticky top-0 z-40 border-b border-slate-200 bg-white/90 backdrop-blur">
        <div className="mx-auto flex w-full max-w-7xl items-center justify-between px-4 py-3 sm:px-6 lg:px-8">
          <div>
            <h1 className="text-lg font-extrabold tracking-tight text-ink sm:text-xl">PPIDeclarePilot Dashboard</h1>
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={onRunAgain}
              disabled={isRunning || !lastRunEntries}
              className="rounded-xl border border-cyan-300 bg-cyan-50 px-3 py-2 text-sm font-semibold text-cyan-700 transition hover:bg-cyan-100 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {isRunning ? 'Running...' : 'Run again'}
            </button>
            <button
              type="button"
              onClick={() => setShowForm((v) => !v)}
              className="rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm font-semibold text-slate-700 transition hover:bg-slate-50"
            >
              New run
            </button>
          </div>
        </div>
      </header>

      <div className="mx-auto w-full max-w-7xl px-4 py-6 sm:px-6 lg:px-8" ref={contentRef}>
        {showForm ? (
          <section className="panel p-5 sm:p-6">
            <form className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3" onSubmit={onSubmit}>
              <div>
                <label className="field-label">Event log (.xes)</label>
                <input className="field-input" type="file" name="event_log_file" accept=".xes" required />
              </div>

              <div>
                <label className="field-label">Provider</label>
                <select className="field-input" name="provider" defaultValue="openai" required>
                  {PROVIDERS.map((p) => (
                    <option key={p.value} value={p.value}>
                      {p.label}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="field-label">Model</label>
                <input className="field-input" type="text" name="model" placeholder="e.g. gpt-5.2" required />
              </div>

              <div>
                <label className="field-label">Base URL (optional)</label>
                <input className="field-input" type="text" name="base_url" placeholder="http://localhost:1234/v1" />
              </div>

              <div>
                <label className="field-label">API Key (optional)</label>
                <input className="field-input" type="password" name="api_key" />
              </div>

              <div className="md:col-span-2 xl:col-span-3 rounded-xl border border-slate-200 bg-slate-50/70 p-4">
                <p className="text-sm font-semibold text-slate-700">Period time grouping</p>
                <div className="mt-3 grid grid-cols-1 gap-4 md:grid-cols-3">
                  <div>
                    <label className="field-label">Quantity</label>
                    <input className="field-input" type="number" name="period_quantity" min="1" defaultValue="1" />
                  </div>
                  <div>
                    <label className="field-label">Unit</label>
                    <select className="field-input" name="period_unit" defaultValue="month">
                      <option value="week">Week(s)</option>
                      <option value="month">Month(s)</option>
                      <option value="year">Year(s)</option>
                    </select>
                  </div>
                </div>
              </div>

              <details className="md:col-span-2 xl:col-span-3 rounded-xl border border-slate-200 bg-white p-3">
                <summary className="cursor-pointer text-sm font-semibold text-slate-700">Advanced parameters</summary>
                <div className="mt-3 grid grid-cols-1 gap-4 md:grid-cols-2">
                  <div>
                    <label className="field-label">Min support (optional)</label>
                    <input className="field-input" type="number" name="min_support" step="0.01" min="0" max="1" placeholder="0.8" />
                    <p className="mt-1 text-xs text-slate-500">
                      Controls how frequent a behavior must be in your process to be considered.
                      Higher values keep only very common patterns; lower values include rarer behaviors.
                    </p>
                  </div>
                  <div>
                    <label className="field-label">Coefficient of variance threshold (optional)</label>
                    <input className="field-input" type="number" name="cv_threshold" step="0.01" min="0" placeholder="0.2" />
                    <p className="mt-1 text-xs text-slate-500">
                      Coefficient of variation threshold used to identify indicators with very low change over time.
                      If a KPI barely changes, it may be less relevant for monitoring.
                    </p>
                  </div>
                  <div className="md:col-span-2">
                    <label className="inline-flex items-center gap-2 text-sm font-semibold text-slate-700">
                      <input type="checkbox" name="use_attributes" defaultChecked />
                      Use event attributes during KPI generation
                    </label>
                  </div>
                </div>
              </details>

              <div className="md:col-span-2 xl:col-span-3">
                <button
                  type="submit"
                  disabled={isRunning}
                  className="inline-flex items-center rounded-xl bg-gradient-to-r from-teal to-aqua px-5 py-2.5 text-sm font-bold text-white shadow-md transition hover:from-teal/90 hover:to-aqua/90 disabled:cursor-wait disabled:opacity-70"
                >
                  {isRunning ? 'Running pipeline...' : 'Run pipeline'}
                </button>
              </div>
            </form>
          </section>
        ) : null}

        {status ? (
          <div className="mt-4 rounded-xl border border-blue-200 bg-blue-50 px-4 py-3 text-sm font-medium text-blue-800">{status}</div>
        ) : null}
        {error ? (
          <div className="mt-4 rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm font-medium text-rose-800">{error}</div>
        ) : null}

        {data ? (
          <>
            <section className="panel mt-5 p-4">
              <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
                <div className="rounded-xl border border-slate-200 bg-white p-3">
                  <p className="text-xs uppercase tracking-wide text-slate-500">KPIs shown</p>
                  <p className="mt-1 text-xl font-bold text-slate-900">{summary.shownKPIs}</p>
                  <p className="text-xs text-slate-500">of {summary.totalKPIs}</p>
                </div>
                <div className="rounded-xl border border-slate-200 bg-white p-3">
                  <p className="text-xs uppercase tracking-wide text-slate-500">Goals shown</p>
                  <p className="mt-1 text-xl font-bold text-slate-900">{summary.shownGoals}</p>
                  <p className="text-xs text-slate-500">of {summary.totalGoals}</p>
                </div>
                <div className="rounded-xl border border-slate-200 bg-white p-3">
                  <p className="text-xs uppercase tracking-wide text-slate-500">Period</p>
                  <p className="mt-1 text-lg font-bold text-slate-900">{formatPeriodLabel(summary.period)}</p>
                </div>
                <div className="rounded-xl border border-slate-200 bg-white p-3">
                  <p className="text-xs uppercase tracking-wide text-slate-500">Minimum variability threshold</p>
                  <p className="mt-1 text-lg font-bold text-slate-900">{summary.cvThreshold ?? '-'}</p>
                </div>
              </div>
            </section>

            <div className="mt-5 grid grid-cols-1 gap-5 lg:grid-cols-[260px_1fr]">
              <aside className="space-y-3 lg:sticky lg:top-20 lg:h-fit">
                <div className="panel p-3">
                  <p className="mb-2 px-2 text-xs font-semibold uppercase tracking-wide text-slate-500">Display controls</p>
                  <div className="space-y-2">
                    <div className="inline-flex w-full rounded-xl border border-slate-300 bg-slate-100 p-1 text-sm">
                      <button
                        type="button"
                        onClick={() => setValueMode('grouped')}
                        className={`w-1/2 rounded-lg px-3 py-1.5 font-semibold transition ${valueMode === 'grouped' ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-600'}`}
                      >
                        Grouped
                      </button>
                      <button
                        type="button"
                        onClick={() => setValueMode('single')}
                        className={`w-1/2 rounded-lg px-3 py-1.5 font-semibold transition ${valueMode === 'single' ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-600'}`}
                      >
                        Single
                      </button>
                    </div>
                    <button
                      type="button"
                      onClick={() => setHideLowCv((v) => !v)}
                      className={`inline-flex w-full items-center gap-2 rounded-full border px-3 py-1.5 text-sm font-semibold transition ${hideLowCv ? 'border-cyan-300 bg-cyan-50 text-cyan-700' : 'border-slate-300 bg-white text-slate-600'}`}
                    >
                      <span className={`h-2.5 w-2.5 rounded-full ${hideLowCv ? 'bg-cyan-500' : 'bg-slate-300'}`} />
                      Keep KPIs above variability threshold
                    </button>
                    <button
                      type="button"
                      onClick={() => setDebugMode((v) => !v)}
                      className={`inline-flex w-full items-center gap-2 rounded-full border px-3 py-1.5 text-sm font-semibold transition ${debugMode ? 'border-amber-300 bg-amber-50 text-amber-700' : 'border-slate-300 bg-white text-slate-600'}`}
                    >
                      <span className={`h-2.5 w-2.5 rounded-full ${debugMode ? 'bg-amber-500' : 'bg-slate-300'}`} />
                      Debug mode
                    </button>
                    <label className="inline-flex w-full items-center gap-2 text-sm text-slate-700">
                      Sort
                      <select className="field-input !w-full !py-1.5" value={sortKpis} onChange={(e) => setSortKpis(e.target.value)}>
                        <option value="id_asc">KPI ID (A-Z)</option>
                        <option value="id_desc">KPI ID (Z-A)</option>
                        <option value="cv_desc">CV (high-low)</option>
                        <option value="cv_asc">CV (low-high)</option>
                        <option value="status">Status</option>
                      </select>
                    </label>
                  </div>
                </div>

                <div className="panel h-fit p-3">
                  <p className="mb-2 px-2 text-xs font-semibold uppercase tracking-wide text-slate-500">Business goals</p>
                  <nav className="space-y-1">
                    {filteredGoals.map((goal) => {
                      const id = goalSectionId(goal.goal_name)
                      const isActive = activeGoal === id
                      return (
                        <div key={id} className={`rounded-lg ${isActive ? 'bg-cyan-50 ring-1 ring-cyan-200' : 'hover:bg-slate-100'}`}>
                          <div className="flex items-center gap-1 px-1.5 py-1">
                            <button
                              type="button"
                              onClick={() => scrollToGoal(goal.goal_name)}
                              className={`min-w-0 flex-1 rounded-md px-1 py-1 text-left text-sm transition ${isActive ? 'text-cyan-800' : 'text-slate-700'}`}
                            >
                              <span className="line-clamp-2 pr-2">{goal.goal_name}</span>
                            </button>
                            <div className="relative">
                              <button
                                type="button"
                                onClick={(e) => {
                                  e.stopPropagation()
                                  setOpenGoalTooltip((prev) => (prev === id ? '' : id))
                                }}
                                className="inline-flex h-5 w-5 items-center justify-center rounded-full border border-slate-300 text-[10px] text-slate-500 hover:bg-white"
                                aria-label="Show goal description"
                              >
                                i
                              </button>
                              {openGoalTooltip === id && goal.goal_description ? (
                                <div
                                  className="absolute right-0 top-6 z-30 w-60 rounded-lg border border-slate-200 bg-white p-2 text-xs text-slate-700 shadow-lg"
                                  onClick={(e) => e.stopPropagation()}
                                >
                                  {goal.goal_description}
                                </div>
                              ) : null}
                            </div>
                            <span className={`rounded-full px-2 py-0.5 text-xs font-semibold ${isActive ? 'bg-cyan-100 text-cyan-700' : 'bg-slate-200 text-slate-700'}`}>
                              {goal.shown_computable_kpis}/{goal.total_kpis}
                            </span>
                          </div>
                        </div>
                      )
                    })}
                  </nav>
                </div>
              </aside>

              <section className="space-y-5">
                {filteredGoals.map((goal) => (
                  <article id={goalSectionId(goal.goal_name)} key={goal.goal_name} className="panel p-4 sm:p-5 scroll-mt-24">
                    <div className="mb-3 flex items-start justify-between gap-3">
                      <div>
                        <h2 className="text-xl font-bold text-ink">{goal.goal_name}</h2>
                        <p className="mt-1 text-sm text-slate-600">{goal.goal_description || 'No goal description available.'}</p>
                      </div>
                      <span className="badge border-cyan-300 bg-cyan-50 text-cyan-700">
                        {goal.shown_computable_kpis}/{goal.total_kpis} KPIs
                      </span>
                    </div>

                    <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
                      {goal.shown_kpis.map((kpi) => {
                        const cv = toCvNumber(kpi.coefficient_of_variation)
                        const panelId = sanitizeId(kpi.kpi_id)
                        const isDuration = kpiHasDurationValues(kpi)
                        return (
                          <div key={`${goal.goal_name}-${panelId}`} className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
                            <div className="mb-2 flex items-center justify-between gap-2">
                              <h3 className="text-base font-extrabold text-slate-900">{kpi.kpi_id}</h3>
                              {kpi.execution_status === 'failed' ? (
                                <span className={statusBadgeClass(kpi.execution_status)}>failed</span>
                              ) : null}
                            </div>

                            <div className="mb-3 flex flex-wrap gap-2">
                              <span className={cvBadgeClass(cv)}>CV: {cv == null ? '-' : cv.toFixed(4)}</span>
                            </div>

                            <div className="space-y-2 text-sm text-slate-700">
                              <p>
                                <span className="font-semibold text-slate-900">Definition:</span>{' '}
                                {kpi.human_readable_definition || 'Not available'}
                                {kpi.human_readable_warning ? (
                                  <span className="ml-2 text-xs text-slate-400">AI generated</span>
                                ) : null}
                              </p>
                              <p>
                                <span className="font-semibold text-slate-900">Objective:</span> {kpi.objective || 'Not available'}
                              </p>
                              <p>
                                <span className="font-semibold text-slate-900">Rationale:</span>{' '}
                                {kpi.category_rationale || 'Not available'}
                              </p>
                            </div>

                            <div className="mt-3">
                              {valueMode === 'single' ? (
                                <SingleValueMetric value={kpi.single_value} isDuration={isDuration} />
                              ) : (
                                <GroupedValueChart
                                  groupedValues={kpi.grouped_values}
                                  timeGrouperFreq={data.meta.time_grouper_freq}
                                  singleValue={kpi.single_value}
                                  isDuration={isDuration}
                                />
                              )}
                            </div>

                            <details className="mt-3 rounded-xl border border-slate-200 bg-slate-50 p-3 text-sm">
                              <summary className="cursor-pointer font-semibold text-slate-800">Technical details</summary>
                              <pre className="mt-2 overflow-x-auto whitespace-pre-wrap break-all rounded-lg bg-slate-900 p-3 text-xs text-slate-100">
                                {kpi.kpi_metric_str || 'Not available'}
                              </pre>
                              {kpi.execution_error ? (
                                <pre className="mt-2 overflow-x-auto whitespace-pre-wrap break-all rounded-lg bg-rose-900/90 p-3 text-xs text-rose-100">
                                  {kpi.execution_error}
                                </pre>
                              ) : null}
                            </details>
                          </div>
                        )
                      })}
                    </div>
                  </article>
                ))}
              </section>
            </div>
          </>
        ) : null}
      </div>
    </div>
  )
}

export default App
