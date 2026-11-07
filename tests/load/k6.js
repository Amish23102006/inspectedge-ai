// k6 load test for POST /inspect. Runs inside the cluster as a Job (tests/load/k6-job.yaml)
// so traffic goes through the Service and is spread over all ready pods.
//
// Env:
//   TARGET_URL  base URL (default: in-cluster Service DNS name)
//   PROFILE     "ramp" (default; drives HPA scale-up) or "steady" (low constant rate,
//               used as background traffic by scripts/k8s/recovery-test.sh)
//   MAX_VUS     ramp peak virtual users (default 12)
//   RATE        steady requests/second (default 2)
//   DURATION    steady duration (default 4m)
import http from 'k6/http';
import { check } from 'k6';

const BASE = __ENV.TARGET_URL || 'http://inspectedge-api.inspectedge.svc.cluster.local';
const PROFILE = __ENV.PROFILE || 'ramp';
const MAX_VUS = parseInt(__ENV.MAX_VUS || '12', 10);
const RATE = parseInt(__ENV.RATE || '2', 10);
const DURATION = __ENV.DURATION || '4m';
const image = open('/load/sample.png', 'b');

const scenarios = {
  ramp: {
    executor: 'ramping-vus',
    startVUs: 0,
    stages: [
      { duration: '1m', target: Math.max(1, Math.round(MAX_VUS / 3)) },
      { duration: '3m', target: MAX_VUS },
      { duration: '2m', target: MAX_VUS },
      { duration: '1m', target: 0 },
    ],
    gracefulRampDown: '30s',
  },
  steady: {
    executor: 'constant-arrival-rate',
    rate: RATE,
    timeUnit: '1s',
    duration: DURATION,
    preAllocatedVUs: Math.max(4, RATE * 4),
    maxVUs: Math.max(20, RATE * 10),
  },
};

export const options = {
  scenarios: { [PROFILE]: scenarios[PROFILE] },
  summaryTrendStats: ['avg', 'min', 'med', 'p(90)', 'p(95)', 'p(99)', 'max'],
  thresholds: {
    // Reported, not used to fail the Job: the point is to record what happened.
    http_req_failed: [{ threshold: 'rate<0.01', abortOnFail: false }],
  },
};

export default function () {
  const res = http.post(
    `${BASE}/inspect`,
    { file: http.file(image, 'sample.png', 'image/png') },
    { timeout: '30s', tags: { route: '/inspect' } },
  );
  check(res, {
    'status is 200': (r) => r.status === 200,
    'has decision': (r) => r.status === 200 && r.json('decision') !== undefined,
  });
}

function ms(metric, stat) {
  const v = metric && metric.values ? metric.values[stat] : undefined;
  return v === undefined ? 'n/a' : v.toFixed(1);
}

export function handleSummary(data) {
  const m = data.metrics;
  const reqs = m.http_reqs ? m.http_reqs.values : { count: 0, rate: 0 };
  const failed = m.http_req_failed ? m.http_req_failed.values.rate : 0;
  const result = {
    profile: PROFILE,
    requests: reqs.count,
    throughput_rps: Number(reqs.rate.toFixed(2)),
    error_rate: Number(failed.toFixed(4)),
    latency_ms: {
      p50: ms(m.http_req_duration, 'med'),
      p90: ms(m.http_req_duration, 'p(90)'),
      p95: ms(m.http_req_duration, 'p(95)'),
      p99: ms(m.http_req_duration, 'p(99)'),
      max: ms(m.http_req_duration, 'max'),
    },
  };
  const text =
    `\n=== InspectEdge load test (${PROFILE}) ===\n` +
    `requests:    ${result.requests}\n` +
    `throughput:  ${result.throughput_rps} req/s\n` +
    `error rate:  ${(failed * 100).toFixed(2)} %\n` +
    `latency ms:  p50 ${result.latency_ms.p50} | p95 ${result.latency_ms.p95} | ` +
    `p99 ${result.latency_ms.p99} | max ${result.latency_ms.max}\n` +
    `K6_SUMMARY_JSON ${JSON.stringify(result)}\n`;
  return { stdout: text };
}
