import jax.numpy as jnp
from jax import jit


@jit
def generate_squeezed_light_jax(dim, squeezing_factor):
    """
    Generate a squeezed vacuum state using JAX.

    Args:
        dim (int): Hilbert space dimension.
        squeezing_factor (float): Squeezing parameter.

    Returns:
        jnp.array: Squeezed state as a JAX tensor.
    """
    print("DEBUG: Generating squeezed vacuum state...")
    vacuum = jnp.zeros(dim)
    vacuum = vacuum.at[0].set(1)  # Set the vacuum state |0⟩

    # Squeeze operator approximation
    squeeze_operator = jnp.diag(jnp.exp(-squeezing_factor * jnp.arange(dim)))
    squeezed_state = squeeze_operator @ vacuum
    return squeezed_state


@jit
def apply_loss_jax(state, efficiency):
    """
    Apply photon loss using JAX.

    Args:
        state (jnp.array): Quantum state as a JAX tensor.
        efficiency (float): Detection efficiency (0 < efficiency <= 1).

    Returns:
        jnp.array: State after applying loss.
    """
    print("DEBUG: Applying photon loss...")
    dim = int(state.shape[0])  # Ensure dimension is concrete
    loss_operator = jnp.sqrt(efficiency) * jnp.eye(dim)
    return loss_operator @ state


@jit
def simulate_faraday_rotation_jax(state, magnetic_field, interaction_time):
    """
    Simulate Faraday rotation using JAX.

    Args:
        state (jnp.array): Quantum state as a JAX tensor.
        magnetic_field (float): Magnetic field strength (T).
        interaction_time (float): Interaction time.

    Returns:
        jnp.array: Rotated state.
    """
    print("DEBUG: Simulating Faraday rotation...")
    dim = int(state.shape[0])  # Ensure dimension is concrete
    rotation_angle = magnetic_field * interaction_time
    rotation_operator = jnp.diag(jnp.exp(1j * rotation_angle * jnp.arange(dim)))
    rotated_state = rotation_operator @ state
    return rotated_state
