async function run() {
  const fidget = await import("./pkg")!;
  onmessage = function (e) {
    postMessage(e);
  };
}
run();
