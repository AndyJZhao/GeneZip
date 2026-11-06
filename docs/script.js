const dialog = document.getElementById('figure-dialog');
const enlargedFigure = document.getElementById('enlarged-figure');
let activeFigureLink;

document.querySelectorAll('[data-lightbox]').forEach(link => {
  link.addEventListener('click', event => {
    if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    activeFigureLink = link;
    enlargedFigure.src = link.href;
    enlargedFigure.alt = link.querySelector('img').alt;
    document.getElementById('figure-caption').textContent = link.dataset.caption;
    dialog.showModal();
  });
});

document.getElementById('close-figure').addEventListener('click', () => dialog.close());
dialog.addEventListener('click', event => {
  if (event.target === dialog) {
    const rect = dialog.getBoundingClientRect();
    if (event.clientX < rect.left || event.clientX > rect.right ||
        event.clientY < rect.top || event.clientY > rect.bottom) dialog.close();
  }
});
dialog.addEventListener('close', () => activeFigureLink?.focus());

const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
document.querySelectorAll('.routing-animation').forEach(figure => {
  const image = figure.querySelector('img[data-animation]');
  const button = figure.querySelector('.animation-toggle');
  function showStill(still) {
    image.src = still ? image.dataset.still : image.dataset.animation;
    figure.querySelector('[data-lightbox]').href = image.src;
    button.setAttribute('aria-pressed', String(still));
    button.textContent = still ? 'Replay animation' : 'Show final checkpoint';
  }
  figure.querySelectorAll('[data-species]').forEach(choice => {
    choice.addEventListener('click', () => {
      if (choice.getAttribute('aria-pressed') === 'true') return;
      figure.querySelectorAll('[data-species]').forEach(item => {
        item.setAttribute('aria-pressed', String(item === choice));
      });
      image.dataset.animation = choice.dataset.animation;
      image.dataset.still = choice.dataset.still;
      const isRepeat = figure.id === 'repeat-routing';
      image.alt = isRepeat
        ? `${choice.dataset.species} DNA: orange tandem-repeat and blue non-repeat routing boundaries during training, then a two-stage compression animation with compression ratios and synchronized local detail.`
        : `${choice.dataset.species} DNA: region-colored routing during training, fixed smooth token-density curves revealed checkpoint by checkpoint, then a two-stage compression illustration with compression ratios.`;
      figure.querySelector('[data-lightbox]').setAttribute('aria-label', `Enlarge ${choice.dataset.species} ${isRepeat ? 'tandem-repeat compression' : 'compression training'} animation`);
      showStill(button.getAttribute('aria-pressed') === 'true');
    });
  });
  showStill(reducedMotion.matches);
  button.addEventListener('click', () => showStill(button.getAttribute('aria-pressed') !== 'true'));
  reducedMotion.addEventListener('change', event => showStill(event.matches));
});

document.getElementById('copy-citation').addEventListener('click', async () => {
  const bibtex = document.getElementById('bibtex');
  const status = document.getElementById('copy-status');
  try {
    await navigator.clipboard.writeText(bibtex.textContent);
    status.textContent = 'BibTeX copied.';
  } catch {
    const range = document.createRange();
    range.selectNodeContents(bibtex);
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    status.textContent = 'Citation selected. Press Ctrl+C or ⌘C to copy.';
  }
});
