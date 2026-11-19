// xgbe_bench: on-device test firmware for xgb-edge (Seeed XIAO nRF52840, Arduino core).
//
// The host sends feature rows over USB serial; the board runs the model under
// test on each row and returns the predicted class, the summed margins and
// the inference time measured on the device.  The model is linked in through
// the small adapter in mut.h / mut.c, which harness/device.py generates.
//
// Frame (both directions, little endian):
//   0xA5 0x5A  type(1)  seq(1)  len(2)  payload(len)  crc16(2, CCITT over type..payload)
// Host -> board:  'I' info request
//                 'P' predict: reps(1) nrows(1) rows(nrows * row_bytes)
//                 'S' stack probe: one row
// Board -> host:  'i' info, 'p' results, 's' stack bytes, 'e' error code

#include <Arduino.h>
#include <Adafruit_TinyUSB.h>  // USB CDC Serial on the Seeed/Adafruit nRF52 core
#include <string.h>

extern "C" {
#include "mut.h"
}

#define PROTO_VERSION 2
#define MAX_PAYLOAD 4096
#define ROW_BYTES (MUT_NUM_FEATURES * MUT_FEATURE_BYTES)
#define RESULT_BYTES (12 + 4 * MUT_NUM_MARGINS)

static uint8_t rx[MAX_PAYLOAD + 8] __attribute__((aligned(4)));
static uint8_t tx[MAX_PAYLOAD + 8] __attribute__((aligned(4)));
static uint32_t rowbuf[(ROW_BYTES + 3) / 4 + 1];
static uint32_t marginbuf[MUT_NUM_MARGINS + 1];

static uint16_t crc16(const uint8_t *p, size_t n, uint16_t crc) {
  while (n--) {
    crc ^= (uint16_t)(*p++) << 8;
    for (int i = 0; i < 8; i++) crc = (crc & 0x8000) ? (uint16_t)((crc << 1) ^ 0x1021) : (uint16_t)(crc << 1);
  }
  return crc;
}

// 16 MHz hardware timer and the 64 MHz core cycle counter, both free running.
static void timing_begin() {
  NRF_CLOCK->EVENTS_HFCLKSTARTED = 0;
  NRF_CLOCK->TASKS_HFCLKSTART = 1;  // crystal oscillator (already on while USB is active)
  NRF_TIMER4->TASKS_STOP = 1;
  NRF_TIMER4->MODE = TIMER_MODE_MODE_Timer;
  NRF_TIMER4->BITMODE = TIMER_BITMODE_BITMODE_32Bit;
  NRF_TIMER4->PRESCALER = 0;
  NRF_TIMER4->TASKS_CLEAR = 1;
  NRF_TIMER4->TASKS_START = 1;
  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
  DWT->CYCCNT = 0;
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

static inline uint32_t timer_ticks() {
  NRF_TIMER4->TASKS_CAPTURE[0] = 1;
  return NRF_TIMER4->CC[0];
}

static bool read_exact(uint8_t *dst, size_t n, uint32_t timeout_ms) {
  size_t got = 0;
  uint32_t t0 = millis();
  while (got < n) {
    int a = Serial.available();
    if (a > 0) {
      size_t want = n - got;
      if ((size_t)a < want) want = (size_t)a;
      got += Serial.readBytes((char *)dst + got, want);
      t0 = millis();
    } else {
      if (millis() - t0 > timeout_ms) return false;
      yield();
    }
  }
  return true;
}

static void send_frame(uint8_t type, uint8_t seq, const uint8_t *payload, uint16_t len) {
  uint8_t head[6] = {0xA5, 0x5A, type, seq, (uint8_t)(len & 0xFF), (uint8_t)(len >> 8)};
  uint16_t crc = crc16(head + 2, 4, 0xFFFF);
  crc = crc16(payload, len, crc);
  uint8_t tail[2] = {(uint8_t)(crc & 0xFF), (uint8_t)(crc >> 8)};
  Serial.write(head, 6);
  size_t off = 0;
  while (off < len) {
    size_t k = len - off;
    if (k > 256) k = 256;
    Serial.write(payload + off, k);
    off += k;
  }
  Serial.write(tail, 2);
  Serial.flush();
}

static void send_error(uint8_t seq, uint8_t code) { send_frame('e', seq, &code, 1); }

static void put32(uint8_t *p, uint32_t v) { memcpy(p, &v, 4); }

static void do_info(uint8_t seq) {
  uint8_t *p = tx;
  memcpy(p, "XGBE", 4);
  p[4] = PROTO_VERSION;
  p[5] = MUT_FEATURE_BYTES;
  p[6] = MUT_MARGIN_IS_FLOAT;
  p[7] = 0;
  uint16_t nf = MUT_NUM_FEATURES, nm = MUT_NUM_MARGINS, mp = MAX_PAYLOAD;
  memcpy(p + 8, &nf, 2);
  memcpy(p + 10, &nm, 2);
  put32(p + 12, (uint32_t)MUT_BUILD_ID);
  memcpy(p + 16, &mp, 2);
  // cost of an empty measurement, so the host can subtract it
  uint32_t c0 = DWT->CYCCNT, t0 = timer_ticks(), t1 = timer_ticks(), c1 = DWT->CYCCNT;
  put32(p + 18, t1 - t0);
  put32(p + 22, c1 - c0);
  send_frame('i', seq, tx, 26);
}

static void do_predict(uint8_t seq, uint16_t len) {
  if (len < 2) { send_error(seq, 2); return; }
  uint8_t reps = rx[0], nrows = rx[1];
  if (reps == 0) reps = 1;
  if ((uint32_t)nrows * ROW_BYTES + 2 != len || (uint32_t)nrows * RESULT_BYTES > MAX_PAYLOAD) {
    send_error(seq, 3);
    return;
  }
  uint8_t *out = tx;
  for (uint8_t r = 0; r < nrows; r++) {
    memcpy(rowbuf, rx + 2 + (uint32_t)r * ROW_BYTES, ROW_BYTES);
    uint32_t best_t = 0xFFFFFFFFu, best_c = 0xFFFFFFFFu;
    int cls = 0;
    for (uint8_t k = 0; k < reps; k++) {
      uint32_t t0 = timer_ticks();
      uint32_t c0 = DWT->CYCCNT;
      cls = mut_predict(rowbuf, marginbuf);
      uint32_t c1 = DWT->CYCCNT;
      uint32_t t1 = timer_ticks();
      if (t1 - t0 < best_t) best_t = t1 - t0;
      if (c1 - c0 < best_c) best_c = c1 - c0;
    }
    put32(out, (uint32_t)cls);
    put32(out + 4, best_t);
    put32(out + 8, best_c);
    memcpy(out + 12, marginbuf, 4 * MUT_NUM_MARGINS);
    out += RESULT_BYTES;
  }
  send_frame('p', seq, tx, (uint16_t)(out - tx));
}

// Stack used by one inference: paint the free stack below the stack pointer, run the
// model, and see how deep it wrote.  Interrupts are off so nothing else touches the stack.
static uint32_t __attribute__((noinline)) stack_probe() {
  const uint32_t span = 3072;
  uintptr_t sp;
  noInterrupts();
  __asm volatile("mov %0, sp" : "=r"(sp));
  volatile uint8_t *top = (volatile uint8_t *)sp;
  for (uint32_t i = 1; i <= span; i++) top[-(int32_t)i] = 0xC5;
  volatile int cls = mut_predict(rowbuf, marginbuf);
  (void)cls;
  uint32_t used = 0;
  for (uint32_t i = span; i >= 1; i--) {
    if (top[-(int32_t)i] != 0xC5) { used = i; break; }
  }
  interrupts();
  return used;
}

static void do_stack(uint8_t seq, uint16_t len) {
  if (len != ROW_BYTES) { send_error(seq, 3); return; }
  memcpy(rowbuf, rx, ROW_BYTES);
  put32(tx, stack_probe());
  send_frame('s', seq, tx, 4);
}

void setup() {
  Serial.begin(115200);
  timing_begin();
}

void loop() {
  uint8_t b;
  if (!Serial.available()) { delay(1); return; }
  if (!read_exact(&b, 1, 10) || b != 0xA5) return;
  if (!read_exact(&b, 1, 100) || b != 0x5A) return;
  uint8_t head[4];
  if (!read_exact(head, 4, 200)) return;
  uint16_t len = (uint16_t)(head[2] | (head[3] << 8));
  if (len > MAX_PAYLOAD) { send_error(head[1], 1); return; }
  if (!read_exact(rx, len, 2000)) { send_error(head[1], 4); return; }
  uint8_t tail[2];
  if (!read_exact(tail, 2, 200)) { send_error(head[1], 4); return; }
  uint16_t crc = crc16(head, 4, 0xFFFF);
  crc = crc16(rx, len, crc);
  if (crc != (uint16_t)(tail[0] | (tail[1] << 8))) { send_error(head[1], 5); return; }
  switch (head[0]) {
    case 'I': do_info(head[1]); break;
    case 'P': do_predict(head[1], len); break;
    case 'S': do_stack(head[1], len); break;
    default: send_error(head[1], 6); break;
  }
}
